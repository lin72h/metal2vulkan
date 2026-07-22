//! Cross-subsystem regression fixtures for retained-SPIR-V access and CFG normalization.

use crate::passes::access::*;
use crate::passes::air_calls::conversions::is_i1_type_token;
use crate::passes::cfg_repair::{
    fix_merge_placement, repair_continue_selection_merge_targets,
    repair_loop_continue_external_predecessors, repair_loop_continue_pass_through_targets,
    repair_phi_predecessor_edges,
};
use crate::passes::f32_to_f16_bits;
use crate::spirv_module::Operand;
use crate::spirv_module::{Block, Function, Instruction, Module, ModuleHeader};
use spirv::{Decoration, FunctionControl, Op, SelectionControl, StorageClass};

// --- Idempotence harness (refactor T5) -------------------------------------------------------
// Every fixup pass in the `transform_with_options` "2d/2e" blocks must be a no-op on its own output
// (the driver even hand-unrolls some of them): running it twice must not change the module. These
// helpers generalize the one existing hand-written idempotence check (decorate_ptr_...). Each
// single-pass fixture below runs its pass through `run_idempotent` instead of calling it once, so a
// pass that mutates its own output turns the fixture red — a finding to journal, not paper over.

/// A comparable snapshot of the mutable state a lowering pass can touch: the function bodies, the
/// global type/constant table, the decorations, and the pass-appended `new_globals` staging vec.
/// Header-agnostic (hand-built fixtures don't all set a full header), so we compare Debug rather
/// than `assemble()`.
fn ctx_state(ctx: &crate::passes::Ctx) -> String {
    format!(
        "{:?}",
        (
            &ctx.module.functions,
            &ctx.module.types_global_values,
            &ctx.module.annotations,
            &ctx.new_globals,
        )
    )
}

/// Run `pass` once (its real transform), then a second time, and assert the second run changed
/// nothing. Leaves `ctx` in the (idempotent) post-run state so a fixture's existing assertions
/// still hold.
fn run_idempotent(ctx: &mut crate::passes::Ctx, mut pass: impl FnMut(&mut crate::passes::Ctx)) {
    pass(ctx);
    let after_first = ctx_state(ctx);
    pass(ctx);
    assert_eq!(
        after_first,
        ctx_state(ctx),
        "pass is not idempotent: a second run mutated the module"
    );
}
// ---------------------------------------------------------------------------------------------

// `is_i1_type_token` matches a real `i1` bool TYPE token (scalar or vector) but NOT the wider ints
// `i16`/`i12` whose spelling contains "i1" (the `air.convert.u.v2i16.f.v2f32` mis-lowering bug). It
// is applied per dst/src token: e.g. for `air.convert.u.i1.f.f32`, dst=`i1` (bool), src=`f32`.
#[test]
fn i1_token_detection() {
    assert!(is_i1_type_token("i1")); // scalar bool dst
    assert!(is_i1_type_token("v3i1")); // bool vector src
    assert!(!is_i1_type_token("v2i16")); // ushort vector
    assert!(!is_i1_type_token("i12")); // 12-bit int
    assert!(!is_i1_type_token("f32"));
    assert!(!is_i1_type_token("i32"));
}

// The f32->f16 bit encoder produces the canonical binary16 patterns for the values we synthesize.
#[test]
fn f16_encoding() {
    assert_eq!(f32_to_f16_bits(0.0), 0x0000);
    assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
}

// A raw WORD index (byte/4) that survives onto a typed-struct binding (the dual-use
// FullChainConversionParams family) is remapped to the MEMBER index at the same byte offset. The
// access chain `%buf %uint_0 %uint_10` over `{ struct_inner, uint, uint, uint }` over-runs the
// 8-member inner struct; word 10 = byte 40 = member 7 (a uint at Offset 40), so the trailing index
// becomes a constant 7 — byte-identical, and the chain validates.
#[test]
fn remap_word_index_to_struct_member_rewrites_oob_word_to_member() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    // Types and globals.
    let uint = 1; // OpTypeInt 32 0
    let float = 2;
    let v4float = 3;
    let struct_inner = 4; // { v4float, uint x7 } -- 8 members
    let struct_outer = 5; // { struct_inner, uint, uint, uint }
    let ptr_sb_outer = 6;
    let ptr_sb_uint = 7;
    let buf = 8;
    let uint_0 = 9;
    let uint_10 = 10;
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypeVector,
            None,
            Some(v4float),
            vec![Operand::IdRef(float), Operand::LiteralBit32(4)],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_inner),
            vec![
                Operand::IdRef(v4float),
                Operand::IdRef(uint),
                Operand::IdRef(uint),
                Operand::IdRef(uint),
                Operand::IdRef(uint),
                Operand::IdRef(uint),
                Operand::IdRef(uint),
                Operand::IdRef(uint),
            ],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_outer),
            vec![
                Operand::IdRef(struct_inner),
                Operand::IdRef(uint),
                Operand::IdRef(uint),
                Operand::IdRef(uint),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_outer),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(struct_outer),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_uint),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(uint),
            ],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr_sb_outer),
            Some(buf),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_0),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_10),
            vec![Operand::LiteralBit32(10)],
        ),
    ];

    // Member byte offsets for the inner struct: v4float@0, then uints at 16,20,24,28,32,36,40.
    let offsets = [0u32, 16, 20, 24, 28, 32, 36, 40];
    for (m, off) in offsets.iter().enumerate() {
        module.annotations.push(Instruction::new(
            Op::MemberDecorate,
            None,
            None,
            vec![
                Operand::IdRef(struct_inner),
                Operand::LiteralBit32(m as u32),
                Operand::Decoration(Decoration::Offset),
                Operand::LiteralBit32(*off),
            ],
        ));
    }

    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![Instruction::new(
                Op::InBoundsAccessChain,
                Some(ptr_sb_uint),
                Some(60),
                vec![
                    Operand::IdRef(buf),
                    Operand::IdRef(uint_0),
                    Operand::IdRef(uint_10),
                ],
            )],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| remap_word_index_to_struct_member(c, 0));

    let chain = &ctx.module.functions[0].blocks[0].instructions[0];
    let Some(Operand::IdRef(last)) = chain.operands.last() else {
        panic!("chain lost its trailing index");
    };
    // The trailing index now points at a uint constant of value 7 (member index, not word 10).
    let cval = ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .find(|g| g.result_id == Some(*last) && g.class.opcode == Op::Constant)
        .and_then(|g| match g.operands.first() {
            Some(Operand::LiteralBit32(v)) => Some(*v),
            _ => None,
        });
    assert_eq!(
        cval,
        Some(7),
        "word index 10 should remap to member index 7"
    );
}

// A raw WORD index that OVERFLOWS the inner sub-struct it descended into is remapped to the SIBLING
// top-level member of the OUTER buffer struct at the same absolute byte offset. The chain
// `%buf %uint_0 %uint_14` descends into outer member 0 (an 8-member inner struct, 44 bytes) then
// applies a buffer-relative flat word index 14 = byte 56, which over-runs the inner struct but lands
// EXACTLY on outer member 3 (a uint at Offset 56). The whole index list collapses to `%buf %uint_3` —
// byte-identical, and the chain validates.
#[test]
fn remap_overflow_word_index_collapses_to_outer_sibling_member() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let uint = 1;
    let float = 2;
    let v4float = 3;
    let struct_inner = 4; // { v4float, uint x7 } -- 8 members, 44 bytes
    let struct_outer = 5; // { struct_inner@0, uint@48, uint@52, uint@56 }
    let ptr_sb_outer = 6;
    let ptr_sb_uint = 7;
    let buf = 8;
    let uint_0 = 9;
    let uint_14 = 10;
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypeVector,
            None,
            Some(v4float),
            vec![Operand::IdRef(float), Operand::LiteralBit32(4)],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_inner),
            vec![
                Operand::IdRef(v4float),
                Operand::IdRef(uint),
                Operand::IdRef(uint),
                Operand::IdRef(uint),
                Operand::IdRef(uint),
                Operand::IdRef(uint),
                Operand::IdRef(uint),
                Operand::IdRef(uint),
            ],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_outer),
            vec![
                Operand::IdRef(struct_inner),
                Operand::IdRef(uint),
                Operand::IdRef(uint),
                Operand::IdRef(uint),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_outer),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(struct_outer),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_uint),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(uint),
            ],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr_sb_outer),
            Some(buf),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_0),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_14),
            vec![Operand::LiteralBit32(14)],
        ),
    ];

    // Inner struct member offsets (v4float@0, uints 16..40); outer member offsets 0, 48, 52, 56.
    let inner_offsets = [0u32, 16, 20, 24, 28, 32, 36, 40];
    for (m, off) in inner_offsets.iter().enumerate() {
        module.annotations.push(Instruction::new(
            Op::MemberDecorate,
            None,
            None,
            vec![
                Operand::IdRef(struct_inner),
                Operand::LiteralBit32(m as u32),
                Operand::Decoration(Decoration::Offset),
                Operand::LiteralBit32(*off),
            ],
        ));
    }
    let outer_offsets = [0u32, 48, 52, 56];
    for (m, off) in outer_offsets.iter().enumerate() {
        module.annotations.push(Instruction::new(
            Op::MemberDecorate,
            None,
            None,
            vec![
                Operand::IdRef(struct_outer),
                Operand::LiteralBit32(m as u32),
                Operand::Decoration(Decoration::Offset),
                Operand::LiteralBit32(*off),
            ],
        ));
    }

    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![Instruction::new(
                Op::InBoundsAccessChain,
                Some(ptr_sb_uint),
                Some(60),
                vec![
                    Operand::IdRef(buf),
                    Operand::IdRef(uint_0),
                    Operand::IdRef(uint_14),
                ],
            )],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| {
        remap_overflow_word_index_to_outer_member(c, 0)
    });

    let chain = &ctx.module.functions[0].blocks[0].instructions[0];
    // The whole index list collapsed to a single trailing index (base + one member index).
    assert_eq!(
        chain.operands.len(),
        2,
        "chain should collapse to base + single member index"
    );
    let Some(Operand::IdRef(last)) = chain.operands.last() else {
        panic!("chain lost its trailing index");
    };
    let cval = ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .find(|g| g.result_id == Some(*last) && g.class.opcode == Op::Constant)
        .and_then(|g| match g.operands.first() {
            Some(Operand::LiteralBit32(v)) => Some(*v),
            _ => None,
        });
    assert_eq!(
        cval,
        Some(3),
        "byte 56 (word 14) should collapse to outer member index 3"
    );
}

// A Workgroup `[3 x {uint,uint}]` (struct stride 2 words) addressed by a flat-WORD index
// `1 + idx*2` feeding a `uint` OpStore is remodeled as a flat `[6 x uint]` array: the variable's
// pointee becomes the uint array and the chain result becomes a `_ptr_Workgroup_uint`, so the uint
// store matches directly.
#[test]
fn remodel_workgroup_flatword_aggregate_flattens_to_uint_array() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let uint = 1;
    let elem = 2; // { uint, uint } -- 8 bytes = 2 words
    let arr = 3; // [3 x elem]
    let ptr_wg_arr = 4;
    let ptr_wg_elem = 5;
    let wgvar = 6;
    let uint_1 = 7;
    let uint_2 = 8;
    let arr_len = 9; // const 3 for the array length
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(elem),
            vec![Operand::IdRef(uint), Operand::IdRef(uint)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(arr_len),
            vec![Operand::LiteralBit32(3)],
        ),
        Instruction::new(
            Op::TypeArray,
            None,
            Some(arr),
            vec![Operand::IdRef(elem), Operand::IdRef(arr_len)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_wg_arr),
            vec![
                Operand::StorageClass(StorageClass::Workgroup),
                Operand::IdRef(arr),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_wg_elem),
            vec![
                Operand::StorageClass(StorageClass::Workgroup),
                Operand::IdRef(elem),
            ],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr_wg_arr),
            Some(wgvar),
            vec![Operand::StorageClass(StorageClass::Workgroup)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_1),
            vec![Operand::LiteralBit32(1)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_2),
            vec![Operand::LiteralBit32(2)],
        ),
    ];

    // Body: %mul = idx*2; %word = 1 + %mul; %chain = AC wgvar %word; store the uint param through it.
    let param = 20;
    let mul = 21;
    let word = 22;
    let chain = 23;
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![Instruction::new(
            Op::FunctionParameter,
            Some(uint),
            Some(param),
            vec![],
        )],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::IMul,
                    Some(uint),
                    Some(mul),
                    vec![Operand::IdRef(param), Operand::IdRef(uint_2)],
                ),
                Instruction::new(
                    Op::IAdd,
                    Some(uint),
                    Some(word),
                    vec![Operand::IdRef(uint_1), Operand::IdRef(mul)],
                ),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_wg_elem),
                    Some(chain),
                    vec![Operand::IdRef(wgvar), Operand::IdRef(word)],
                ),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(chain), Operand::IdRef(param)],
                ),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    remodel_workgroup_flatword_aggregate(&mut ctx, 0);

    let all_globals: Vec<&Instruction> = ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .collect();
    let pointee_of = |ptr: u32| -> Option<u32> {
        all_globals.iter().find_map(|g| {
            if g.result_id == Some(ptr) && g.class.opcode == Op::TypePointer {
                match g.operands.get(1) {
                    Some(Operand::IdRef(p)) => Some(*p),
                    _ => None,
                }
            } else {
                None
            }
        })
    };
    // The chain result is now a Workgroup pointer to a plain uint scalar.
    let chain_inst = &ctx.module.functions[0].blocks[0].instructions[2];
    let chain_ptr = chain_inst.result_type.expect("chain has a result type");
    assert_eq!(
        pointee_of(chain_ptr),
        Some(uint),
        "chain result must point at uint after remodel"
    );
    // The variable now points at a `[6 x uint]` array (3 elements * 2 words).
    let var_inst = all_globals
        .iter()
        .find(|g| g.class.opcode == Op::Variable && g.result_id == Some(wgvar))
        .expect("variable still present");
    let var_arr = pointee_of(var_inst.result_type.unwrap()).expect("var points at an array");
    let arr_def = all_globals
        .iter()
        .find(|g| g.result_id == Some(var_arr) && g.class.opcode == Op::TypeArray)
        .expect("variable pointee is an array");
    let Some(Operand::IdRef(arr_elem)) = arr_def.operands.first() else {
        panic!("array missing element type");
    };
    assert_eq!(*arr_elem, uint, "flat array element must be uint");
    let Some(Operand::IdRef(len_c)) = arr_def.operands.get(1) else {
        panic!("array missing length");
    };
    let len_val = all_globals
        .iter()
        .find(|g| g.result_id == Some(*len_c) && g.class.opcode == Op::Constant)
        .and_then(|g| match g.operands.first() {
            Some(Operand::LiteralBit32(v)) => Some(*v),
            _ => None,
        });
    assert_eq!(len_val, Some(6), "flat array length must be 3*2 = 6 words");
}

// A Workgroup `array<float, 3>` atomically reduced as uint (the Metal float-as-signed-int min idiom):
// `%c = AC float %wg %idx; %cu = OpBitcast uint-ptr %c; OpAtomicSMin %uint %cu ...` plus a plain
// `OpLoad %float %c`. The pass retypes the array to `array<uint, 3>`, drops the pointer bitcast,
// repoints the atomic at the uint chain, and splits the float load into load-uint + value-bitcast.
#[test]
fn remodel_workgroup_floatarray_atomic_as_uint_retypes_and_repoints() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let uint = 1;
    let float = 2;
    let uint_ptr_wg = 3; // _ptr_Workgroup_uint (the float-as-uint bitcast target; pre-exists)
    let float_arr = 4; // [3 x float]
    let ptr_wg_arr = 5;
    let ptr_wg_float = 6;
    let wgvar = 7;
    let arr_len = 8; // const 3
    let idx = 9; // const 0 index
    let scope = 10; // const 1 (Device)
    let sem = 11; // const 0
    let val = 12; // const uint operand for the atomic
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(arr_len),
            vec![Operand::LiteralBit32(3)],
        ),
        Instruction::new(
            Op::TypeArray,
            None,
            Some(float_arr),
            vec![Operand::IdRef(float), Operand::IdRef(arr_len)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_wg_arr),
            vec![
                Operand::StorageClass(StorageClass::Workgroup),
                Operand::IdRef(float_arr),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_wg_float),
            vec![
                Operand::StorageClass(StorageClass::Workgroup),
                Operand::IdRef(float),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(uint_ptr_wg),
            vec![
                Operand::StorageClass(StorageClass::Workgroup),
                Operand::IdRef(uint),
            ],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr_wg_arr),
            Some(wgvar),
            vec![Operand::StorageClass(StorageClass::Workgroup)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(idx),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(scope),
            vec![Operand::LiteralBit32(1)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(sem),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(val),
            vec![Operand::LiteralBit32(7)],
        ),
    ];

    let chain = 20;
    let cu = 21;
    let atomic = 22;
    let loaded = 23;
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_wg_float),
                    Some(chain),
                    vec![Operand::IdRef(wgvar), Operand::IdRef(idx)],
                ),
                Instruction::new(
                    Op::Bitcast,
                    Some(uint_ptr_wg),
                    Some(cu),
                    vec![Operand::IdRef(chain)],
                ),
                Instruction::new(
                    Op::AtomicSMin,
                    Some(uint),
                    Some(atomic),
                    vec![
                        Operand::IdRef(cu),
                        Operand::IdRef(scope),
                        Operand::IdRef(sem),
                        Operand::IdRef(val),
                    ],
                ),
                Instruction::new(
                    Op::Load,
                    Some(float),
                    Some(loaded),
                    vec![Operand::IdRef(chain)],
                ),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    remodel_workgroup_floatarray_atomic_as_uint(&mut ctx, 0).unwrap();

    let all_globals: Vec<&Instruction> = ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .collect();
    let pointee_of = |ptr: u32| -> Option<u32> {
        all_globals.iter().find_map(|g| {
            if g.result_id == Some(ptr) && g.class.opcode == Op::TypePointer {
                match g.operands.get(1) {
                    Some(Operand::IdRef(p)) => Some(*p),
                    _ => None,
                }
            } else {
                None
            }
        })
    };
    // The variable now points at a `[3 x uint]` array.
    let var_ptr = all_globals
        .iter()
        .find(|g| g.result_id == Some(wgvar))
        .and_then(|g| g.result_type)
        .expect("var has a result type");
    let new_arr = pointee_of(var_ptr).expect("var points at an array");
    let new_arr_def = all_globals
        .iter()
        .find(|g| g.result_id == Some(new_arr))
        .expect("array type def");
    assert_eq!(new_arr_def.class.opcode, Op::TypeArray);
    assert_eq!(
        new_arr_def.operands.first(),
        Some(&Operand::IdRef(uint)),
        "array element must be uint after retype"
    );

    let body = &ctx.module.functions[0].blocks[0].instructions;
    // The chain now points at a uint element.
    let chain_inst = body
        .iter()
        .find(|i| i.result_id == Some(chain))
        .expect("chain survives");
    assert_eq!(
        pointee_of(chain_inst.result_type.unwrap()),
        Some(uint),
        "chain must point at uint after retype"
    );
    // The pointer bitcast is gone, and the atomic points straight at the chain.
    assert!(
        !body.iter().any(|i| i.result_id == Some(cu)),
        "the pointer OpBitcast must be removed"
    );
    let atomic_inst = body
        .iter()
        .find(|i| i.result_id == Some(atomic))
        .expect("atomic survives");
    assert_eq!(
        atomic_inst.operands.first(),
        Some(&Operand::IdRef(chain)),
        "atomic must point at the uint chain, not the dropped bitcast"
    );
    // The float load is split into a uint load + a value bitcast preserving the original result id.
    let load_bitcast = body
        .iter()
        .find(|i| i.class.opcode == Op::Bitcast && i.result_id == Some(loaded))
        .expect("the float load became a value bitcast");
    assert_eq!(load_bitcast.result_type, Some(float));
    let uint_load_src = match load_bitcast.operands.first() {
        Some(Operand::IdRef(s)) => *s,
        _ => panic!("bitcast source"),
    };
    let uint_load = body
        .iter()
        .find(|i| i.class.opcode == Op::Load && i.result_id == Some(uint_load_src))
        .expect("a uint load feeds the value bitcast");
    assert_eq!(uint_load.result_type, Some(uint));
    assert_eq!(uint_load.operands.first(), Some(&Operand::IdRef(chain)));
}

// The SIGNED-int variant: an integer histogram living in a threadgroup `float[3]` array, accumulated
// by `air.atomic.local.add.s.i32` (a SIGNED `OpAtomicIAdd %int` through a `_ptr_Workgroup_int` bitcast)
// and read back with a plain `OpLoad %int`, initialized by `OpStore %float_0`. The pass must DETECT the
// signed int reinterpret type (not assume uint), retype the array to `[3 x int]`, repoint the atomic AND
// the plain int load at the int chain, and split the float store into bitcast(float->int)+store.
#[test]
fn remodel_workgroup_floatarray_atomic_as_signed_int_add_retypes_and_repoints() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let uint = 1; // for the array length / scope / sem constants
    let float = 2;
    let int_ptr_wg = 3; // _ptr_Workgroup_int (the float-as-SIGNED-int bitcast target; pre-exists)
    let float_arr = 4; // [3 x float]
    let ptr_wg_arr = 5;
    let ptr_wg_float = 6;
    let wgvar = 7;
    let arr_len = 8; // const 3
    let idx = 9; // const 0 index
    let scope = 10; // const 1
    let sem = 11; // const 0
    let val = 12; // const int operand for the atomic add
    let int_s = 13; // OpTypeInt 32 1 (signed)
    let f0 = 14; // OpConstant float 0
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(int_s),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(1)],
        ),
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(arr_len),
            vec![Operand::LiteralBit32(3)],
        ),
        Instruction::new(
            Op::TypeArray,
            None,
            Some(float_arr),
            vec![Operand::IdRef(float), Operand::IdRef(arr_len)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_wg_arr),
            vec![
                Operand::StorageClass(StorageClass::Workgroup),
                Operand::IdRef(float_arr),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_wg_float),
            vec![
                Operand::StorageClass(StorageClass::Workgroup),
                Operand::IdRef(float),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(int_ptr_wg),
            vec![
                Operand::StorageClass(StorageClass::Workgroup),
                Operand::IdRef(int_s),
            ],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr_wg_arr),
            Some(wgvar),
            vec![Operand::StorageClass(StorageClass::Workgroup)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(idx),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(scope),
            vec![Operand::LiteralBit32(1)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(sem),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(int_s),
            Some(val),
            vec![Operand::LiteralBit32(7)],
        ),
        Instruction::new(
            Op::Constant,
            Some(float),
            Some(f0),
            vec![Operand::LiteralBit32(0)],
        ),
    ];

    let chain = 20;
    let cu = 21;
    let atomic = 22;
    let histload = 23;
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_wg_float),
                    Some(chain),
                    vec![Operand::IdRef(wgvar), Operand::IdRef(idx)],
                ),
                Instruction::new(
                    Op::Bitcast,
                    Some(int_ptr_wg),
                    Some(cu),
                    vec![Operand::IdRef(chain)],
                ),
                Instruction::new(
                    Op::AtomicIAdd,
                    Some(int_s),
                    Some(atomic),
                    vec![
                        Operand::IdRef(cu),
                        Operand::IdRef(scope),
                        Operand::IdRef(sem),
                        Operand::IdRef(val),
                    ],
                ),
                // Histogram read-back: a plain signed-int load through the same bitcast pointer.
                Instruction::new(
                    Op::Load,
                    Some(int_s),
                    Some(histload),
                    vec![Operand::IdRef(cu)],
                ),
                // Initialization: a plain float store of 0.0 through the float chain.
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(chain), Operand::IdRef(f0)],
                ),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    remodel_workgroup_floatarray_atomic_as_uint(&mut ctx, 0).unwrap();

    let all_globals: Vec<&Instruction> = ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .collect();
    let pointee_of = |ptr: u32| -> Option<u32> {
        all_globals.iter().find_map(|g| {
            if g.result_id == Some(ptr) && g.class.opcode == Op::TypePointer {
                match g.operands.get(1) {
                    Some(Operand::IdRef(p)) => Some(*p),
                    _ => None,
                }
            } else {
                None
            }
        })
    };
    // The variable now points at a `[3 x int]` array (the SIGNED int, not uint).
    let var_ptr = all_globals
        .iter()
        .find(|g| g.result_id == Some(wgvar))
        .and_then(|g| g.result_type)
        .expect("var has a result type");
    let new_arr = pointee_of(var_ptr).expect("var points at an array");
    let new_arr_def = all_globals
        .iter()
        .find(|g| g.result_id == Some(new_arr))
        .expect("array type def");
    assert_eq!(new_arr_def.class.opcode, Op::TypeArray);
    assert_eq!(
        new_arr_def.operands.first(),
        Some(&Operand::IdRef(int_s)),
        "array element must be the SIGNED int after retype"
    );

    let body = &ctx.module.functions[0].blocks[0].instructions;
    // The pointer bitcast is gone; the atomic AND the plain int load both point straight at the chain.
    assert!(
        !body.iter().any(|i| i.result_id == Some(cu)),
        "the pointer OpBitcast must be removed"
    );
    let atomic_inst = body
        .iter()
        .find(|i| i.result_id == Some(atomic))
        .expect("atomic survives");
    assert_eq!(
        atomic_inst.operands.first(),
        Some(&Operand::IdRef(chain)),
        "atomic must point at the int chain"
    );
    let hist_inst = body
        .iter()
        .find(|i| i.result_id == Some(histload))
        .expect("histogram load survives");
    assert_eq!(hist_inst.class.opcode, Op::Load);
    assert_eq!(hist_inst.result_type, Some(int_s));
    assert_eq!(
        hist_inst.operands.first(),
        Some(&Operand::IdRef(chain)),
        "the plain int load must be repointed natively at the int chain"
    );
    // The float store became bitcast(float_0 -> int) + store of the int temporary.
    let store = body
        .iter()
        .find(|i| i.class.opcode == Op::Store)
        .expect("the store survives");
    let stored = match store.operands.get(1) {
        Some(Operand::IdRef(s)) => *s,
        _ => panic!("store object"),
    };
    let store_bitcast = body
        .iter()
        .find(|i| i.class.opcode == Op::Bitcast && i.result_id == Some(stored))
        .expect("the float store object became an int value bitcast");
    assert_eq!(store_bitcast.result_type, Some(int_s));
    assert_eq!(store_bitcast.operands.first(), Some(&Operand::IdRef(f0)));
}

// A thread-local `i64` slot reused as `float[2]` is over-indexed by `AC %_ptr_Function_float %slot
// %uint_1` (illegal: indexing a scalar). The pass re-expresses the dependent `OpLoad %float` as a
// shift/truncate/bitcast of the whole slot (element 1 = the high 32 bits), removing the chain.
#[test]
fn rewrite_scalar_slot_array_overindex_lowers_union_element_load() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let uint = 1;
    let float = 2;
    let ulong = 3;
    let ptr_func_ulong = 4;
    let ptr_func_float = 5;
    let uint_1 = 6;
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(ulong),
            vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_func_ulong),
            vec![
                Operand::StorageClass(StorageClass::Function),
                Operand::IdRef(ulong),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_func_float),
            vec![
                Operand::StorageClass(StorageClass::Function),
                Operand::IdRef(float),
            ],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_1),
            vec![Operand::LiteralBit32(1)],
        ),
    ];

    let slot = 60;
    let chain = 61;
    let loaded = 62;
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(float),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::Variable,
                    Some(ptr_func_ulong),
                    Some(slot),
                    vec![Operand::StorageClass(StorageClass::Function)],
                ),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_func_float),
                    Some(chain),
                    vec![Operand::IdRef(slot), Operand::IdRef(uint_1)],
                ),
                Instruction::new(
                    Op::Load,
                    Some(float),
                    Some(loaded),
                    vec![Operand::IdRef(chain)],
                ),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| {
        rewrite_scalar_slot_array_overindex(c, 0).unwrap();
    });

    let body = &ctx.module.functions[0].blocks[0].instructions;
    assert!(
        !body
            .iter()
            .any(|i| matches!(i.class.opcode, Op::InBoundsAccessChain | Op::AccessChain)),
        "the scalar over-index chain should be removed"
    );
    assert!(
        body.iter().any(|i| i.class.opcode == Op::ShiftRightLogical),
        "element 1 should be read via a right shift of the whole slot"
    );
    assert!(
        body.iter().any(|i| i.class.opcode == Op::UConvert),
        "the shifted slot should be truncated to 32 bits"
    );
    // The original load result id is preserved (its consumers are unchanged).
    assert!(
        body.iter().any(|i| i.result_id == Some(loaded)),
        "the load result id should survive the rewrite"
    );
}

#[test]
fn rewrite_scalar_pointer_arithmetic_promotes_self_typed_inbounds_to_ptr_access_chain() {
    // A StorageBuffer `OpInBoundsAccessChain` whose RESULT pointer type equals its BASE pointer type
    // is scalar pointer arithmetic — indexing a `float*` as if it were an array — illegal as an
    // InBoundsAccessChain (there is no aggregate to index into). The pass promotes it to
    // `OpPtrAccessChain`, which strides by the index, for the storage classes that permit it
    // (StorageBuffer/Workgroup/PhysicalStorageBuffer). A base whose type differs from the result must
    // be left alone (genuine aggregate descent), which the negative case below pins.
    let float = 1;
    let ptr_sb_float = 2;
    let uint = 3;
    let idx = 4;
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_float),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(float),
            ],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(idx),
            vec![Operand::LiteralBit32(5)],
        ),
    ];

    // `%base` is a `float*` parameter; the chain result is the SAME `float*` — the trigger shape.
    let base = 50;
    let chain = 60;
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(float),
            Some(40),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(41),
            ],
        )),
        parameters: vec![Instruction::new(
            Op::FunctionParameter,
            Some(ptr_sb_float),
            Some(base),
            vec![],
        )],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![Instruction::new(
                Op::InBoundsAccessChain,
                Some(ptr_sb_float),
                Some(chain),
                vec![Operand::IdRef(base), Operand::IdRef(idx)],
            )],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| {
        rewrite_scalar_pointer_arithmetic_access_chains(c, 0)
    });

    let body = &ctx.module.functions[0].blocks[0].instructions;
    assert!(
        !body
            .iter()
            .any(|i| i.class.opcode == Op::InBoundsAccessChain),
        "the self-typed scalar InBoundsAccessChain should be promoted"
    );
    let promoted = body
        .iter()
        .find(|i| i.result_id == Some(chain))
        .expect("the chain result id survives the rewrite");
    assert_eq!(
        promoted.class.opcode,
        Op::PtrAccessChain,
        "promoted to OpPtrAccessChain"
    );
    assert_eq!(
        promoted.result_type,
        Some(ptr_sb_float),
        "result pointer type is preserved"
    );
    assert_eq!(
        promoted.operands,
        vec![Operand::IdRef(base), Operand::IdRef(idx)],
        "base + index operands are preserved"
    );
}

#[test]
fn rewrite_scalar_pointer_arithmetic_leaves_genuine_aggregate_descent_alone() {
    // Negative case: an InBoundsAccessChain whose BASE type differs from its RESULT type is a real
    // aggregate descent (e.g. `float(*)[4]` -> `float*`), NOT scalar pointer arithmetic — it must be
    // left as an InBoundsAccessChain (rewriting it to PtrAccessChain would be a mistyped stride).
    let float = 1;
    let arr4 = 2;
    let ptr_sb_arr = 3;
    let ptr_sb_float = 4;
    let uint = 5;
    let idx = 6;
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(idx),
            vec![Operand::LiteralBit32(4)],
        ),
        Instruction::new(
            Op::TypeArray,
            None,
            Some(arr4),
            vec![Operand::IdRef(float), Operand::IdRef(idx)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_arr),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(arr4),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_float),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(float),
            ],
        ),
    ];

    // `%base` is a `float(*)[4]`; the chain result is `float*` — base type != result type.
    let base = 50;
    let chain = 60;
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(float),
            Some(40),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(41),
            ],
        )),
        parameters: vec![Instruction::new(
            Op::FunctionParameter,
            Some(ptr_sb_arr),
            Some(base),
            vec![],
        )],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![Instruction::new(
                Op::InBoundsAccessChain,
                Some(ptr_sb_float),
                Some(chain),
                vec![Operand::IdRef(base), Operand::IdRef(idx)],
            )],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });

    let mut ctx = crate::passes::Ctx::new(module);
    rewrite_scalar_pointer_arithmetic_access_chains(&mut ctx, 0);

    let chain_inst = ctx.module.functions[0].blocks[0]
        .instructions
        .iter()
        .find(|i| i.result_id == Some(chain))
        .expect("the chain survives");
    assert_eq!(
        chain_inst.class.opcode,
        Op::InBoundsAccessChain,
        "genuine aggregate descent must NOT be promoted to OpPtrAccessChain"
    );
}

#[test]
fn rewrite_reinterpret_scalar_loads_same_width_splits_into_declared_load_plus_bitcast() {
    // A StorageBuffer `OpLoad %uint` through a `float*` pointer is a same-width reinterpret: the
    // load's Result Type (uint) differs from the pointer's declared pointee (float), both 32-bit.
    // The pass rewrites it into a load in the DECLARED type followed by an `OpBitcast` to the result
    // type (the original load's result id moves onto the bitcast, so consumers are unchanged).
    let float = 1;
    let uint = 2;
    let ptr_sb_float = 3;
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_float),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(float),
            ],
        ),
    ];

    let ptr = 50; // a `float*` StorageBuffer parameter
    let loaded = 60; // OpLoad %uint through the float pointer — currently invalid
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(40),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(41),
            ],
        )),
        parameters: vec![Instruction::new(
            Op::FunctionParameter,
            Some(ptr_sb_float),
            Some(ptr),
            vec![],
        )],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![Instruction::new(
                Op::Load,
                Some(uint),
                Some(loaded),
                vec![Operand::IdRef(ptr)],
            )],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| rewrite_reinterpret_scalar_loads(c, 0));

    let body = &ctx.module.functions[0].blocks[0].instructions;
    // No load keeps the invalid `uint` result type any more.
    assert!(
        !body
            .iter()
            .any(|i| i.class.opcode == Op::Load && i.result_type == Some(uint)),
        "the mistyped uint load should be gone"
    );
    // The declared-type load is now `OpLoad %float` with a fresh result id.
    let decl_load = body
        .iter()
        .find(|i| i.class.opcode == Op::Load && i.result_type == Some(float))
        .expect("a load in the declared float pointee type is emitted");
    assert_ne!(
        decl_load.result_id,
        Some(loaded),
        "the declared-type load gets a fresh id, not the original load id"
    );
    // The original load's result id survives on an `OpBitcast %uint` fed by that declared load.
    let bitcast = body
        .iter()
        .find(|i| i.result_id == Some(loaded))
        .expect("the original load result id survives");
    assert_eq!(
        bitcast.class.opcode,
        Op::Bitcast,
        "the result id now names the reinterpret bitcast"
    );
    assert_eq!(
        bitcast.result_type,
        Some(uint),
        "bitcast yields the uint value"
    );
    assert_eq!(
        bitcast.operands,
        vec![Operand::IdRef(decl_load.result_id.unwrap())],
        "the bitcast reinterprets the declared-type load"
    );
}

#[test]
fn rewrite_strided_descent_promotes_overindexed_array_chain_to_ptr_access_chain() {
    // A StorageBuffer InBoundsAccessChain over a `[4 x float]*` base with TWO indices over-indexes
    // (walking both indices runs off the scalar element — invalid), but reading the FIRST index as a
    // whole-array stride and DESCENDING with the rest reaches the declared `float*` result. The pass
    // promotes it to OpPtrAccessChain (which strides by index 0, then descends). Negative case: a
    // single-index chain (a valid descent) must be left alone.
    let float = 1;
    let uint = 2;
    let len4 = 3;
    let arr = 4;
    let ptr_sb_arr = 5;
    let ptr_sb_float = 6;
    let idx0 = 7;
    let idx1 = 8;
    let make_module = || {
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(100));
        module.types_global_values = vec![
            Instruction::new(
                Op::TypeFloat,
                None,
                Some(float),
                vec![Operand::LiteralBit32(32)],
            ),
            Instruction::new(
                Op::TypeInt,
                None,
                Some(uint),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            Instruction::new(
                Op::Constant,
                Some(uint),
                Some(len4),
                vec![Operand::LiteralBit32(4)],
            ),
            Instruction::new(
                Op::TypeArray,
                None,
                Some(arr),
                vec![Operand::IdRef(float), Operand::IdRef(len4)],
            ),
            Instruction::new(
                Op::TypePointer,
                None,
                Some(ptr_sb_arr),
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(arr),
                ],
            ),
            Instruction::new(
                Op::TypePointer,
                None,
                Some(ptr_sb_float),
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(float),
                ],
            ),
            Instruction::new(
                Op::Constant,
                Some(uint),
                Some(idx0),
                vec![Operand::LiteralBit32(0)],
            ),
            Instruction::new(
                Op::Constant,
                Some(uint),
                Some(idx1),
                vec![Operand::LiteralBit32(2)],
            ),
        ];
        module
    };

    // Positive: two-index over-index → promoted to OpPtrAccessChain.
    let base = 50;
    let chain = 60;
    let mut module = make_module();
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(float),
            Some(40),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(41),
            ],
        )),
        parameters: vec![Instruction::new(
            Op::FunctionParameter,
            Some(ptr_sb_arr),
            Some(base),
            vec![],
        )],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![Instruction::new(
                Op::InBoundsAccessChain,
                Some(ptr_sb_float),
                Some(chain),
                vec![
                    Operand::IdRef(base),
                    Operand::IdRef(idx0),
                    Operand::IdRef(idx1),
                ],
            )],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });
    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| rewrite_strided_descent_access_chains(c, 0));
    let promoted = ctx.module.functions[0].blocks[0]
        .instructions
        .iter()
        .find(|i| i.result_id == Some(chain))
        .expect("chain survives");
    assert_eq!(
        promoted.class.opcode,
        Op::PtrAccessChain,
        "over-indexed array chain promoted to OpPtrAccessChain"
    );
    assert_eq!(
        promoted.operands,
        vec![
            Operand::IdRef(base),
            Operand::IdRef(idx0),
            Operand::IdRef(idx1),
        ],
        "base + stride + descent operands preserved"
    );

    // Negative: a single-index chain into the array is a valid descent — never promoted.
    let mut module = make_module();
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(float),
            Some(40),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(41),
            ],
        )),
        parameters: vec![Instruction::new(
            Op::FunctionParameter,
            Some(ptr_sb_arr),
            Some(base),
            vec![],
        )],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![Instruction::new(
                Op::InBoundsAccessChain,
                Some(ptr_sb_float),
                Some(chain),
                vec![Operand::IdRef(base), Operand::IdRef(idx1)],
            )],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });
    let mut ctx = crate::passes::Ctx::new(module);
    rewrite_strided_descent_access_chains(&mut ctx, 0);
    let valid = ctx.module.functions[0].blocks[0]
        .instructions
        .iter()
        .find(|i| i.result_id == Some(chain))
        .expect("chain survives");
    assert_eq!(
        valid.class.opcode,
        Op::InBoundsAccessChain,
        "a valid single-index descent must NOT be promoted"
    );
}

#[test]
fn repair_loop_continue_external_predecessors_keeps_preheader_out_of_continue() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(Op::TypeVoid, None, Some(1), vec![]),
        Instruction::new(Op::TypeFunction, None, Some(2), vec![Operand::IdRef(1)]),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(3),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(3),
            Some(4),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(3),
            Some(5),
            vec![Operand::LiteralBit32(1)],
        ),
    ];
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(1),
            Some(10),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(2),
            ],
        )),
        parameters: vec![],
        blocks: vec![
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(20), vec![])),
                instructions: vec![Instruction::new(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(30)],
                )],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(30), vec![])),
                instructions: vec![Instruction::new(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(50)],
                )],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(40), vec![])),
                instructions: vec![
                    Instruction::new(
                        Op::Phi,
                        Some(3),
                        Some(70),
                        vec![Operand::IdRef(71), Operand::IdRef(50)],
                    ),
                    Instruction::new(
                        Op::LoopMerge,
                        None,
                        None,
                        vec![
                            Operand::IdRef(60),
                            Operand::IdRef(50),
                            Operand::LoopControl(spirv::LoopControl::NONE),
                        ],
                    ),
                    Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(80)]),
                ],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(80), vec![])),
                instructions: vec![Instruction::new(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(50)],
                )],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(50), vec![])),
                instructions: vec![
                    Instruction::new(
                        Op::Phi,
                        Some(3),
                        Some(71),
                        vec![
                            Operand::IdRef(4),
                            Operand::IdRef(30),
                            Operand::IdRef(5),
                            Operand::IdRef(80),
                        ],
                    ),
                    Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(40)]),
                ],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(60), vec![])),
                instructions: vec![Instruction::new(Op::Return, None, None, vec![])],
            },
        ],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });

    let mut ctx = crate::passes::Ctx::new(module);
    repair_loop_continue_external_predecessors(&mut ctx, 0);
    let blocks = &ctx.module.functions[0].blocks;
    let preheader = blocks
        .iter()
        .find(|block| block.label.as_ref().and_then(|label| label.result_id) == Some(30))
        .unwrap();
    assert_eq!(
        preheader.instructions.last().unwrap().operands,
        vec![Operand::IdRef(40)]
    );
    let header = blocks
        .iter()
        .find(|block| block.label.as_ref().and_then(|label| label.result_id) == Some(40))
        .unwrap();
    assert_eq!(
        header.instructions[0].operands,
        vec![
            Operand::IdRef(71),
            Operand::IdRef(50),
            Operand::IdRef(4),
            Operand::IdRef(30),
        ]
    );
    let continue_block = blocks
        .iter()
        .find(|block| block.label.as_ref().and_then(|label| label.result_id) == Some(50))
        .unwrap();
    assert_eq!(
        continue_block.instructions[0].operands,
        vec![Operand::IdRef(5), Operand::IdRef(80)]
    );
}

#[test]
fn compose_derived_access_chains_rebases_linear_stream_offsets() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(1),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypeVector,
            None,
            Some(2),
            vec![Operand::IdRef(1), Operand::LiteralBit32(4)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(3),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(2),
            ],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(4),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(4),
            Some(5),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(4),
            Some(6),
            vec![Operand::LiteralBit32(7)],
        ),
        Instruction::new(
            Op::Constant,
            Some(4),
            Some(7),
            vec![Operand::LiteralBit32(1)],
        ),
    ];
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(1),
            Some(10),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(11),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(12), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(3),
                    Some(20),
                    vec![Operand::IdRef(30), Operand::IdRef(5), Operand::IdRef(6)],
                ),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(3),
                    Some(21),
                    vec![Operand::IdRef(20), Operand::IdRef(7)],
                ),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| compose_derived_access_chains(c, 0));

    let inst = ctx.module.functions[0].blocks[0]
        .instructions
        .iter()
        .find(|inst| inst.result_id == Some(21))
        .expect("composed access chain");
    assert_eq!(inst.operands[0], Operand::IdRef(30));
    assert_eq!(inst.operands[1], Operand::IdRef(5));
    let Operand::IdRef(composed) = inst.operands[2] else {
        panic!("composed index should be an id");
    };
    assert_eq!(const_i64_value(&ctx, composed), Some(8));
}

#[test]
fn repair_phi_predecessor_edges_reuses_dominating_incoming_value() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(1),
            Some(10),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(11),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(12), vec![])),
                instructions: vec![
                    Instruction::new(Op::IAdd, Some(1), Some(20), vec![]),
                    Instruction::new(
                        Op::BranchConditional,
                        None,
                        None,
                        vec![Operand::IdRef(30), Operand::IdRef(13), Operand::IdRef(14)],
                    ),
                ],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(14), vec![])),
                instructions: vec![Instruction::new(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(13)],
                )],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(13), vec![])),
                instructions: vec![Instruction::new(
                    Op::Phi,
                    Some(1),
                    Some(21),
                    vec![Operand::IdRef(20), Operand::IdRef(12)],
                )],
            },
        ],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    repair_phi_predecessor_edges(&mut ctx, 0);
    let phi = &ctx.module.functions[0].blocks[2].instructions[0];
    assert_eq!(
        phi.operands,
        vec![
            Operand::IdRef(20),
            Operand::IdRef(12),
            Operand::IdRef(20),
            Operand::IdRef(14),
        ]
    );
}

#[test]
fn repair_phi_predecessor_edges_rewrites_stale_self_backedge_to_continue() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(1),
            Some(10),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(11),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(12), vec![])),
                instructions: vec![
                    Instruction::new(Op::IAdd, Some(1), Some(20), vec![]),
                    Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(13)]),
                ],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(13), vec![])),
                instructions: vec![
                    Instruction::new(
                        Op::Phi,
                        Some(1),
                        Some(21),
                        vec![
                            Operand::IdRef(20),
                            Operand::IdRef(12),
                            Operand::IdRef(22),
                            Operand::IdRef(13),
                        ],
                    ),
                    Instruction::new(Op::IAdd, Some(1), Some(22), vec![Operand::IdRef(21)]),
                    Instruction::new(
                        Op::LoopMerge,
                        None,
                        None,
                        vec![
                            Operand::IdRef(15),
                            Operand::IdRef(14),
                            Operand::LoopControl(spirv::LoopControl::NONE),
                        ],
                    ),
                    Instruction::new(
                        Op::BranchConditional,
                        None,
                        None,
                        vec![Operand::IdRef(30), Operand::IdRef(15), Operand::IdRef(14)],
                    ),
                ],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(14), vec![])),
                instructions: vec![Instruction::new(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(13)],
                )],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(15), vec![])),
                instructions: vec![Instruction::new(Op::Return, None, None, vec![])],
            },
        ],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    repair_phi_predecessor_edges(&mut ctx, 0);
    let phi = &ctx.module.functions[0].blocks[1].instructions[0];
    assert_eq!(
        phi.operands,
        vec![
            Operand::IdRef(20),
            Operand::IdRef(12),
            Operand::IdRef(22),
            Operand::IdRef(14),
        ]
    );
}

#[test]
fn repair_phi_predecessor_edges_drops_stale_extra_predecessor() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(1),
            Some(10),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(11),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(13), vec![])),
                instructions: vec![
                    Instruction::new(
                        Op::Phi,
                        Some(1),
                        Some(21),
                        vec![
                            Operand::IdRef(22),
                            Operand::IdRef(13),
                            Operand::IdRef(23),
                            Operand::IdRef(14),
                        ],
                    ),
                    Instruction::new(Op::Return, None, None, vec![]),
                ],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(14), vec![])),
                instructions: vec![Instruction::new(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(13)],
                )],
            },
        ],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    repair_phi_predecessor_edges(&mut ctx, 0);
    let phi = &ctx.module.functions[0].blocks[0].instructions[0];
    assert_eq!(phi.operands, vec![Operand::IdRef(23), Operand::IdRef(14)]);
}

#[test]
fn repair_continue_selection_merge_targets_inserts_in_loop_merge() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(1),
            Some(10),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(11),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(12), vec![])),
                instructions: vec![
                    Instruction::new(
                        Op::LoopMerge,
                        None,
                        None,
                        vec![
                            Operand::IdRef(13),
                            Operand::IdRef(14),
                            Operand::LoopControl(spirv::LoopControl::NONE),
                        ],
                    ),
                    Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(15)]),
                ],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(15), vec![])),
                instructions: vec![
                    Instruction::new(
                        Op::SelectionMerge,
                        None,
                        None,
                        vec![
                            Operand::IdRef(14),
                            Operand::SelectionControl(spirv::SelectionControl::NONE),
                        ],
                    ),
                    Instruction::new(
                        Op::BranchConditional,
                        None,
                        None,
                        vec![Operand::IdRef(20), Operand::IdRef(14), Operand::IdRef(13)],
                    ),
                ],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(13), vec![])),
                instructions: vec![Instruction::new(Op::Return, None, None, vec![])],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(14), vec![])),
                instructions: vec![Instruction::new(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(12)],
                )],
            },
        ],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    repair_continue_selection_merge_targets(&mut ctx, 0);

    let blocks = &ctx.module.functions[0].blocks;
    let selection = &blocks[1].instructions[0];
    assert_eq!(
        selection.operands.first(),
        Some(&Operand::IdRef(100)),
        "selection merge should be the synthetic in-loop block"
    );
    assert_eq!(blocks[1].instructions[1].operands[2], Operand::IdRef(100));
    assert_eq!(
        blocks[2].label.as_ref().and_then(|label| label.result_id),
        Some(100),
        "synthetic block should be inserted before the loop merge"
    );
    assert_eq!(
        blocks[2].instructions[0].operands.first(),
        Some(&Operand::IdRef(13))
    );
}

#[test]
fn repair_continue_selection_merge_targets_splits_continue_reconvergence() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(1),
            Some(10),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(11),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(12), vec![])),
                instructions: vec![
                    Instruction::new(
                        Op::LoopMerge,
                        None,
                        None,
                        vec![
                            Operand::IdRef(13),
                            Operand::IdRef(14),
                            Operand::LoopControl(spirv::LoopControl::NONE),
                        ],
                    ),
                    Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(15)]),
                ],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(15), vec![])),
                instructions: vec![
                    Instruction::new(
                        Op::SelectionMerge,
                        None,
                        None,
                        vec![
                            Operand::IdRef(14),
                            Operand::SelectionControl(spirv::SelectionControl::NONE),
                        ],
                    ),
                    Instruction::new(
                        Op::BranchConditional,
                        None,
                        None,
                        vec![Operand::IdRef(20), Operand::IdRef(16), Operand::IdRef(17)],
                    ),
                ],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(16), vec![])),
                instructions: vec![Instruction::new(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(17)],
                )],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(17), vec![])),
                instructions: vec![Instruction::new(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(14)],
                )],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(14), vec![])),
                instructions: vec![Instruction::new(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(12)],
                )],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(13), vec![])),
                instructions: vec![Instruction::new(Op::Return, None, None, vec![])],
            },
        ],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    repair_continue_selection_merge_targets(&mut ctx, 0);

    let blocks = &ctx.module.functions[0].blocks;
    assert_eq!(
        blocks[1].instructions[0].operands.first(),
        Some(&Operand::IdRef(100))
    );
    assert_eq!(
        blocks[3].instructions[0].operands.first(),
        Some(&Operand::IdRef(100))
    );
    assert_eq!(
        blocks[4].label.as_ref().and_then(|label| label.result_id),
        Some(100)
    );
    assert_eq!(
        blocks[4].instructions[0].operands.first(),
        Some(&Operand::IdRef(14))
    );
}

#[test]
fn repair_loop_continue_pass_through_targets_uses_real_continue_block() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(1),
            Some(10),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(11),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(12), vec![])),
                instructions: vec![
                    Instruction::new(
                        Op::LoopMerge,
                        None,
                        None,
                        vec![
                            Operand::IdRef(13),
                            Operand::IdRef(14),
                            Operand::LoopControl(spirv::LoopControl::NONE),
                        ],
                    ),
                    Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(15)]),
                ],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(15), vec![])),
                instructions: vec![Instruction::new(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(14)],
                )],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(14), vec![])),
                instructions: vec![Instruction::new(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(16)],
                )],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(16), vec![])),
                instructions: vec![Instruction::new(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(12)],
                )],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(13), vec![])),
                instructions: vec![Instruction::new(Op::Return, None, None, vec![])],
            },
        ],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    repair_loop_continue_pass_through_targets(&mut ctx, 0);
    let loop_merge = &ctx.module.functions[0].blocks[0].instructions[0];
    assert_eq!(loop_merge.operands.get(1), Some(&Operand::IdRef(16)));
}

#[test]
fn hoist_function_variables_moves_them_to_entry_front() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(1),
            Some(10),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(11),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(12), vec![])),
                instructions: vec![
                    Instruction::new(Op::Load, Some(1), Some(20), vec![Operand::IdRef(30)]),
                    Instruction::new(
                        Op::Variable,
                        Some(2),
                        Some(21),
                        vec![Operand::StorageClass(StorageClass::Function)],
                    ),
                ],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(13), vec![])),
                instructions: vec![Instruction::new(
                    Op::Variable,
                    Some(2),
                    Some(22),
                    vec![Operand::StorageClass(StorageClass::Function)],
                )],
            },
        ],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| hoist_function_variables(c, 0));

    let first = &ctx.module.functions[0].blocks[0].instructions;
    assert_eq!(first[0].class.opcode, Op::Variable);
    assert_eq!(first[1].class.opcode, Op::Variable);
    assert_eq!(first[2].class.opcode, Op::Load);
    assert!(ctx.module.functions[0].blocks[1].instructions.is_empty());
}

#[test]
fn lower_private_memory_atomics_uses_plain_load_store() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(1),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(2),
            vec![
                Operand::StorageClass(StorageClass::Private),
                Operand::IdRef(1),
            ],
        ),
        Instruction::new(
            Op::Constant,
            Some(1),
            Some(3),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Variable,
            Some(2),
            Some(4),
            vec![
                Operand::StorageClass(StorageClass::Private),
                Operand::IdRef(3),
            ],
        ),
        Instruction::new(
            Op::Constant,
            Some(1),
            Some(5),
            vec![Operand::LiteralBit32(1)],
        ),
        Instruction::new(
            Op::Constant,
            Some(1),
            Some(6),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(1),
            Some(7),
            vec![Operand::LiteralBit32(0xff)],
        ),
    ];
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(1),
            Some(10),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(11),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(12), vec![])),
            instructions: vec![Instruction::new(
                Op::AtomicAnd,
                Some(1),
                Some(20),
                vec![
                    Operand::IdRef(4),
                    Operand::IdScope(5),
                    Operand::IdMemorySemantics(6),
                    Operand::IdRef(7),
                ],
            )],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| lower_private_memory_atomics(c, 0));

    let insts = &ctx.module.functions[0].blocks[0].instructions;
    assert_eq!(insts.len(), 3);
    assert_eq!(insts[0].class.opcode, Op::Load);
    assert_eq!(insts[0].result_id, Some(20));
    assert_eq!(insts[1].class.opcode, Op::BitwiseAnd);
    assert_eq!(insts[2].class.opcode, Op::Store);
    assert!(!insts.iter().any(|inst| inst.class.opcode == Op::AtomicAnd));
}

// OpPtrAccessChain requires its base pointer TYPE to carry an ArrayStride decoration
// (VUID-StandaloneSpirv-None-10684). decorate_ptr_access_chain_base_strides adds the missing
// `ArrayStride = round_up(sizeof pointee)` to each distinct base pointer type, idempotently.
#[test]
fn decorate_ptr_access_chain_base_strides_adds_stride_per_pointee() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(Op::TypeVoid, None, Some(1), vec![]),
        Instruction::new(Op::TypeFunction, None, Some(2), vec![Operand::IdRef(1)]),
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(3),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(4),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(4),
            Some(5),
            vec![Operand::LiteralBit32(7)],
        ),
        // ptr StorageBuffer float  -> expect ArrayStride 4
        Instruction::new(
            Op::TypePointer,
            None,
            Some(6),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(3),
            ],
        ),
        // <4 x float>
        Instruction::new(
            Op::TypeVector,
            None,
            Some(7),
            vec![Operand::IdRef(3), Operand::LiteralBit32(4)],
        ),
        // ptr StorageBuffer <4 x float>  -> expect ArrayStride 16
        Instruction::new(
            Op::TypePointer,
            None,
            Some(8),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(7),
            ],
        ),
        // `ArrayStride` is an explicit layout decoration and is invalid on Workgroup pointers.
        Instruction::new(
            Op::TypePointer,
            None,
            Some(9),
            vec![
                Operand::StorageClass(StorageClass::Workgroup),
                Operand::IdRef(3),
            ],
        ),
    ];
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(1),
            Some(10),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(2),
            ],
        )),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(20), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::PtrAccessChain,
                    Some(6),
                    Some(30),
                    vec![Operand::IdRef(40), Operand::IdRef(5)],
                ),
                Instruction::new(
                    Op::PtrAccessChain,
                    Some(8),
                    Some(31),
                    vec![Operand::IdRef(41), Operand::IdRef(5)],
                ),
                Instruction::new(
                    Op::PtrAccessChain,
                    Some(9),
                    Some(32),
                    vec![Operand::IdRef(42), Operand::IdRef(5)],
                ),
                Instruction::new(Op::Return, None, None, vec![]),
            ],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });

    let mut ctx = crate::passes::Ctx::new(module);
    decorate_ptr_access_chain_base_strides(&mut ctx);

    let stride_of = |ctx: &crate::passes::Ctx, ty: u32| -> Vec<u32> {
        ctx.module
            .annotations
            .iter()
            .filter(|a| {
                a.class.opcode == Op::Decorate
                    && a.operands.first() == Some(&Operand::IdRef(ty))
                    && a.operands.get(1) == Some(&Operand::Decoration(Decoration::ArrayStride))
            })
            .filter_map(|a| match a.operands.get(2) {
                Some(Operand::LiteralBit32(s)) => Some(*s),
                _ => None,
            })
            .collect()
    };
    assert_eq!(stride_of(&ctx, 6), vec![4], "float pointer stride");
    assert_eq!(stride_of(&ctx, 8), vec![16], "<4 x float> pointer stride");
    assert!(
        stride_of(&ctx, 9).is_empty(),
        "Workgroup pointer types cannot carry explicit layout decorations"
    );

    // Idempotent: a second run must not append a duplicate ArrayStride for the same pointer type.
    decorate_ptr_access_chain_base_strides(&mut ctx);
    assert_eq!(stride_of(&ctx, 6), vec![4], "no duplicate after re-run");
    assert_eq!(stride_of(&ctx, 8), vec![16], "no duplicate after re-run");
}

// A `ushort` load through a struct-member `uchar` pointer that spans two adjacent `uchar` members is
// lowered to a little-endian byte assembly: load each member, widen, shift the high byte by 8, OR.
#[test]
fn lower_cross_member_subword_load_assembles_spanned_bytes() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let uchar = 1; // Int 8 0
    let ushort = 2; // Int 16 0
    let uint = 3; // Int 32 0 (for the uint_0 index constant)
    let struct_inner = 4; // { uchar@0, uchar@1 }
    let struct_outer = 5; // { struct_inner }
    let ptr_sb_outer = 6;
    let ptr_sb_uchar = 7;
    let buf = 8;
    let uint_0 = 9;
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uchar),
            vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(ushort),
            vec![Operand::LiteralBit32(16), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_inner),
            vec![Operand::IdRef(uchar), Operand::IdRef(uchar)],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_outer),
            vec![Operand::IdRef(struct_inner)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_outer),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(struct_outer),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_uchar),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(uchar),
            ],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr_sb_outer),
            Some(buf),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_0),
            vec![Operand::LiteralBit32(0)],
        ),
    ];
    for (m, off) in [0u32, 1].iter().enumerate() {
        module.annotations.push(Instruction::new(
            Op::MemberDecorate,
            None,
            None,
            vec![
                Operand::IdRef(struct_inner),
                Operand::LiteralBit32(m as u32),
                Operand::Decoration(Decoration::Offset),
                Operand::LiteralBit32(*off),
            ],
        ));
    }
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_sb_uchar),
                    Some(60),
                    vec![
                        Operand::IdRef(buf),
                        Operand::IdRef(uint_0),
                        Operand::IdRef(uint_0),
                    ],
                ),
                Instruction::new(Op::Load, Some(ushort), Some(61), vec![Operand::IdRef(60)]),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| {
        lower_cross_member_subword_load(c, 0).unwrap();
    });

    let insts = &ctx.module.functions[0].blocks[0].instructions;
    // The mismatched `OpLoad %ushort %60` is gone; a byte assembly (shift + OR) takes its place.
    assert!(
        !insts
            .iter()
            .any(|i| i.class.opcode == Op::Load && i.result_type == Some(ushort)),
        "the ushort load through the uchar pointer must be replaced"
    );
    assert!(
        insts.iter().any(|i| i.class.opcode == Op::ShiftLeftLogical),
        "high byte must be shifted into place"
    );
    assert!(
        insts.iter().any(|i| i.class.opcode == Op::BitwiseOr),
        "the two bytes must be ORed"
    );
    // Two uchar loads (one per spanned member) now feed the assembly.
    assert_eq!(
        insts
            .iter()
            .filter(|i| i.class.opcode == Op::Load && i.result_type == Some(uchar))
            .count(),
        2,
        "one load per spanned member"
    );
    // The original result id is still defined (as the final assembled value).
    assert!(
        insts.iter().any(|i| i.result_id == Some(61)),
        "the load result id must survive as the assembled value"
    );
}

// A `ulong` store through a struct-member `uint` pointer that spans two adjacent `uint` members is
// split into two `uint` stores: the low word to member 0, the high word (shifted right 32) to member 1.
#[test]
fn lower_cross_member_subword_store_splits_into_members() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let uint = 1; // Int 32 0
    let ulong = 2; // Int 64 0
    let struct_inner = 3; // { uint@0, uint@4 }
    let struct_outer = 4; // { struct_inner }
    let ptr_sb_outer = 5;
    let ptr_sb_uint = 6;
    let buf = 7;
    let uint_0 = 8;
    let obj = 9; // the ulong value to store
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(ulong),
            vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_inner),
            vec![Operand::IdRef(uint), Operand::IdRef(uint)],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_outer),
            vec![Operand::IdRef(struct_inner)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_outer),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(struct_outer),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_uint),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(uint),
            ],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr_sb_outer),
            Some(buf),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_0),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(ulong),
            Some(obj),
            vec![Operand::LiteralBit32(7), Operand::LiteralBit32(0)],
        ),
    ];
    for (m, off) in [0u32, 4].iter().enumerate() {
        module.annotations.push(Instruction::new(
            Op::MemberDecorate,
            None,
            None,
            vec![
                Operand::IdRef(struct_inner),
                Operand::LiteralBit32(m as u32),
                Operand::Decoration(Decoration::Offset),
                Operand::LiteralBit32(*off),
            ],
        ));
    }
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_sb_uint),
                    Some(60),
                    vec![
                        Operand::IdRef(buf),
                        Operand::IdRef(uint_0),
                        Operand::IdRef(uint_0),
                    ],
                ),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(60), Operand::IdRef(obj)],
                ),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| lower_cross_member_subword_store(c, 0));

    let insts = &ctx.module.functions[0].blocks[0].instructions;
    // Two uint stores replace the single ulong store, with a right-shift for the high word.
    assert_eq!(
        insts.iter().filter(|i| i.class.opcode == Op::Store).count(),
        2,
        "the ulong store splits into two uint stores"
    );
    assert!(
        insts
            .iter()
            .any(|i| i.class.opcode == Op::ShiftRightLogical),
        "the high word must be shifted down"
    );
    assert!(
        insts.iter().any(|i| i.class.opcode == Op::UConvert),
        "each 32-bit word must be narrowed from the 64-bit object"
    );
    assert!(
        !insts.iter().any(|i| matches!(
            i.operands.get(1),
            Some(Operand::IdRef(id)) if *id == obj
        ) && i.class.opcode == Op::Store),
        "the original ulong object is no longer stored directly"
    );
}

// A dead, currently-invalid access chain (an unused `uchar`-pointer re-indexed to a `ushort` pointer)
// is dropped; a used one and a valid one are kept.
#[test]
fn drop_dead_invalid_access_chains_removes_unused_overindex() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let uchar = 1;
    let ushort = 2;
    let uint = 3;
    let arr = 4; // runtime array of uchar
    let struct_outer = 5; // { arr }
    let ptr_sb_outer = 6;
    let ptr_sb_uchar = 7;
    let ptr_sb_ushort = 8;
    let buf = 9;
    let uint_0 = 10;
    let idx = 11;
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uchar),
            vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(ushort),
            vec![Operand::LiteralBit32(16), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeRuntimeArray,
            None,
            Some(arr),
            vec![Operand::IdRef(uchar)],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_outer),
            vec![Operand::IdRef(arr)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_outer),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(struct_outer),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_uchar),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(uchar),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_ushort),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(ushort),
            ],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr_sb_outer),
            Some(buf),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_0),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(idx),
            vec![Operand::LiteralBit32(2)],
        ),
    ];
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                // A valid uchar element pointer (used below).
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_sb_uchar),
                    Some(60),
                    vec![
                        Operand::IdRef(buf),
                        Operand::IdRef(uint_0),
                        Operand::IdRef(idx),
                    ],
                ),
                // A DEAD, INVALID chain: indexes the scalar uchar pointer to a ushort pointer.
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_sb_ushort),
                    Some(61),
                    vec![Operand::IdRef(60), Operand::IdRef(idx)],
                ),
                // Keep %60 live.
                Instruction::new(Op::Load, Some(uchar), Some(62), vec![Operand::IdRef(60)]),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| drop_dead_invalid_access_chains(c, 0));

    let insts = &ctx.module.functions[0].blocks[0].instructions;
    assert!(
        !insts.iter().any(|i| i.result_id == Some(61)),
        "the dead invalid chain must be dropped"
    );
    assert!(
        insts.iter().any(|i| i.result_id == Some(60)),
        "the valid, used chain must be kept"
    );
    assert!(
        insts.iter().any(|i| i.result_id == Some(62)),
        "the load that keeps %60 live must be kept"
    );
}

// A `Private` `array<half, 2>` written with a `half2` at BYTE offset 4 (via a `uchar`
// `OpPtrAccessChain` off the `&half[0]` base) is lowered to two per-element half stores at element
// indices 2 and 3, and the variable's array is enlarged to `array<half, 4>` so those indices are in
// bounds. The illegal byte chain disappears entirely.
#[test]
fn lower_private_byte_aggregate_reinterpret_splits_v2half_store() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let half = 1;
    let uint = 2;
    let uchar = 3;
    let v2half = 4;
    let arr2 = 5;
    let ptr_priv_arr = 6;
    let ptr_priv_half = 7;
    let ptr_priv_uchar = 8;
    let uint_0 = 9;
    let uint_2 = 10;
    let uint_4 = 11;
    let var = 12;
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(half),
            vec![Operand::LiteralBit32(16)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uchar),
            vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeVector,
            None,
            Some(v2half),
            vec![Operand::IdRef(half), Operand::LiteralBit32(2)],
        ),
        Instruction::new(
            Op::TypeArray,
            None,
            Some(arr2),
            vec![Operand::IdRef(half), Operand::IdRef(uint_2)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_priv_arr),
            vec![
                Operand::StorageClass(StorageClass::Private),
                Operand::IdRef(arr2),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_priv_half),
            vec![
                Operand::StorageClass(StorageClass::Private),
                Operand::IdRef(half),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_priv_uchar),
            vec![
                Operand::StorageClass(StorageClass::Private),
                Operand::IdRef(uchar),
            ],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_0),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_2),
            vec![Operand::LiteralBit32(2)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_4),
            vec![Operand::LiteralBit32(4)],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr_priv_arr),
            Some(var),
            vec![Operand::StorageClass(StorageClass::Private)],
        ),
    ];
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                // The half2 object to store.
                Instruction::new(Op::Undef, Some(v2half), Some(60), vec![]),
                // &half[0].
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_priv_half),
                    Some(61),
                    vec![Operand::IdRef(var), Operand::IdRef(uint_0)],
                ),
                // +4 bytes — an illegal uchar PtrAccessChain off the half element pointer.
                Instruction::new(
                    Op::PtrAccessChain,
                    Some(ptr_priv_uchar),
                    Some(62),
                    vec![Operand::IdRef(61), Operand::IdRef(uint_4)],
                ),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(62), Operand::IdRef(60)],
                ),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    lower_private_byte_aggregate_reinterpret(&mut ctx, 0).unwrap();

    // The illegal byte chain is gone.
    let insts = &ctx.module.functions[0].blocks[0].instructions;
    assert!(
        !insts.iter().any(|i| i.class.opcode == Op::PtrAccessChain),
        "the uchar byte PtrAccessChain must be removed"
    );
    // Two element stores landed at indices 2 and 3 (byte 4 / 2-byte half = element 2).
    let elem_store_indices: Vec<u32> = insts
        .iter()
        .filter(|i| {
            i.class.opcode == Op::InBoundsAccessChain
                && i.operands.first() == Some(&Operand::IdRef(var))
        })
        .filter_map(|i| match i.operands.get(1) {
            Some(Operand::IdRef(c)) => ctx
                .module
                .types_global_values
                .iter()
                .chain(ctx.new_globals.iter())
                .find(|g| g.result_id == Some(*c))
                .and_then(|g| match g.operands.first() {
                    Some(Operand::LiteralBit32(v)) => Some(*v),
                    _ => None,
                }),
            _ => None,
        })
        .collect();
    assert!(
        elem_store_indices.contains(&2) && elem_store_indices.contains(&3),
        "expected per-element accesses at indices 2 and 3, got {elem_store_indices:?}"
    );
    assert_eq!(
        insts.iter().filter(|i| i.class.opcode == Op::Store).count(),
        2,
        "the v2half store must split into two half stores"
    );

    // The variable's array is enlarged to length 4.
    let var_ptr = ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
        .find(|g| g.result_id == Some(var))
        .and_then(|g| g.result_type)
        .expect("variable retyped");
    let new_arr = ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
        .find(|g| g.result_id == Some(var_ptr))
        .and_then(|g| match g.operands.get(1) {
            Some(Operand::IdRef(p)) => Some(*p),
            _ => None,
        })
        .expect("pointer pointee");
    let arr_len_c = ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
        .find(|g| g.result_id == Some(new_arr) && g.class.opcode == Op::TypeArray)
        .and_then(|g| match g.operands.get(1) {
            Some(Operand::IdRef(c)) => Some(*c),
            _ => None,
        })
        .expect("array length");
    let len_val = ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
        .find(|g| g.result_id == Some(arr_len_c))
        .and_then(|g| match g.operands.first() {
            Some(Operand::LiteralBit32(v)) => Some(*v),
            _ => None,
        })
        .expect("length constant");
    assert_eq!(
        len_val, 4,
        "the array must be enlarged to hold element index 3"
    );
}

// A write-only `Private` `uchar` placeholder that is the TARGET of an `OpCopyMemory` from a `struct`
// source is retyped to `_ptr_Private_<struct>` so the copy's pointee types match; its scalar
// initializer is dropped.
#[test]
fn retype_demoted_copymemory_placeholder_matches_source_struct() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let uchar = 1;
    let float = 2;
    let v3float = 3;
    let inner = 4; // struct { v3float }
    let ptr_priv_uchar = 5;
    let ptr_func_inner = 6;
    let null_uchar = 7;
    let placeholder = 8; // Private uchar var (the demoted placeholder)
    let src = 9; // Function struct var (the assembled source)
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uchar),
            vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypeVector,
            None,
            Some(v3float),
            vec![Operand::IdRef(float), Operand::LiteralBit32(3)],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(inner),
            vec![Operand::IdRef(v3float)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_priv_uchar),
            vec![
                Operand::StorageClass(StorageClass::Private),
                Operand::IdRef(uchar),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_func_inner),
            vec![
                Operand::StorageClass(StorageClass::Function),
                Operand::IdRef(inner),
            ],
        ),
        Instruction::new(Op::ConstantNull, Some(uchar), Some(null_uchar), vec![]),
        Instruction::new(
            Op::Variable,
            Some(ptr_priv_uchar),
            Some(placeholder),
            vec![
                Operand::StorageClass(StorageClass::Private),
                Operand::IdRef(null_uchar),
            ],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr_func_inner),
            Some(src),
            vec![Operand::StorageClass(StorageClass::Function)],
        ),
    ];
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uchar),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![Instruction::new(
                Op::CopyMemory,
                None,
                None,
                vec![Operand::IdRef(placeholder), Operand::IdRef(src)],
            )],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    retype_demoted_copymemory_placeholder(&mut ctx, 0);

    // The placeholder is retyped to a Private pointer whose pointee is the source struct.
    let var = ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
        .find(|g| g.result_id == Some(placeholder))
        .expect("placeholder var");
    let new_ptr = var.result_type.expect("retyped");
    let pointee = ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
        .find(|g| g.result_id == Some(new_ptr) && g.class.opcode == Op::TypePointer)
        .and_then(|g| match g.operands.get(1) {
            Some(Operand::IdRef(p)) => Some(*p),
            _ => None,
        })
        .expect("pointer pointee");
    assert_eq!(
        pointee, inner,
        "the placeholder must point to the source struct type"
    );
    // Its scalar initializer is dropped (storage class only).
    assert_eq!(
        var.operands.len(),
        1,
        "the mistyped scalar initializer must be dropped"
    );
}

// A Function `array<float, 9>` whose element-0 pointer `%p = AC %arr %uint_0` is then OVER-indexed
// (`%r = AC %p %idx`) — the two-step lowering of `gep [9 x float], %arr, 0, %idx` that loses array
// provenance — is re-rooted onto the array: `%r = AC %arr %idx` (byte-identical: element 0 + idx is
// element idx). The over-index of the scalar disappears.
#[test]
fn reroot_demoted_array_element_overindex_direct() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let float = 1;
    let uint = 2;
    let uint_9 = 3;
    let arr9 = 4;
    let ptr_func_arr = 5;
    let ptr_func_float = 6;
    let arr_var = 7;
    let uint_0 = 8;
    let idx = 9;
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_9),
            vec![Operand::LiteralBit32(9)],
        ),
        Instruction::new(
            Op::TypeArray,
            None,
            Some(arr9),
            vec![Operand::IdRef(float), Operand::IdRef(uint_9)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_func_arr),
            vec![
                Operand::StorageClass(StorageClass::Function),
                Operand::IdRef(arr9),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_func_float),
            vec![
                Operand::StorageClass(StorageClass::Function),
                Operand::IdRef(float),
            ],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_0),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(idx),
            vec![Operand::LiteralBit32(5)],
        ),
    ];
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(float),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::Variable,
                    Some(ptr_func_arr),
                    Some(arr_var),
                    vec![Operand::StorageClass(StorageClass::Function)],
                ),
                // %p = &arr[0] (element-0 pointer).
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_func_float),
                    Some(60),
                    vec![Operand::IdRef(arr_var), Operand::IdRef(uint_0)],
                ),
                // %r = (&arr[0])[idx] — over-indexes the scalar float.
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_func_float),
                    Some(61),
                    vec![Operand::IdRef(60), Operand::IdRef(idx)],
                ),
                Instruction::new(Op::Load, Some(float), Some(62), vec![Operand::IdRef(61)]),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| reroot_demoted_array_element_overindex(c, 0));

    let r = ctx.module.functions[0].blocks[0]
        .instructions
        .iter()
        .find(|i| i.result_id == Some(61))
        .expect("the over-index chain must still exist");
    assert_eq!(
        r.operands,
        vec![Operand::IdRef(arr_var), Operand::IdRef(idx)],
        "the over-index must be re-rooted onto the array variable with the same index"
    );
}

// The base reaches the array's element-0 through a degenerate `OpPhi` whose arms are both
// `AC %arr %uint_0` — the provenance trace must converge to the single array and re-root the
// over-index onto it.
#[test]
fn reroot_demoted_array_element_overindex_through_phi() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let float = 1;
    let uint = 2;
    let uint_4 = 3;
    let arr4 = 4;
    let ptr_func_arr = 5;
    let ptr_func_float = 6;
    let arr_var = 7;
    let uint_0 = 8;
    let idx = 9;
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_4),
            vec![Operand::LiteralBit32(4)],
        ),
        Instruction::new(
            Op::TypeArray,
            None,
            Some(arr4),
            vec![Operand::IdRef(float), Operand::IdRef(uint_4)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_func_arr),
            vec![
                Operand::StorageClass(StorageClass::Function),
                Operand::IdRef(arr4),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_func_float),
            vec![
                Operand::StorageClass(StorageClass::Function),
                Operand::IdRef(float),
            ],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_0),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(idx),
            vec![Operand::LiteralBit32(2)],
        ),
    ];
    // Two predecessor blocks each define an element-0 pointer; a third block's phi merges them; then
    // the over-index. (A minimal but well-formed CFG for provenance tracing.)
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(float),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(70), vec![])),
                instructions: vec![
                    Instruction::new(
                        Op::Variable,
                        Some(ptr_func_arr),
                        Some(arr_var),
                        vec![Operand::StorageClass(StorageClass::Function)],
                    ),
                    Instruction::new(
                        Op::InBoundsAccessChain,
                        Some(ptr_func_float),
                        Some(80),
                        vec![Operand::IdRef(arr_var), Operand::IdRef(uint_0)],
                    ),
                    Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(72)]),
                ],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(71), vec![])),
                instructions: vec![
                    Instruction::new(
                        Op::InBoundsAccessChain,
                        Some(ptr_func_float),
                        Some(81),
                        vec![Operand::IdRef(arr_var), Operand::IdRef(uint_0)],
                    ),
                    Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(72)]),
                ],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(72), vec![])),
                instructions: vec![
                    Instruction::new(
                        Op::Phi,
                        Some(ptr_func_float),
                        Some(82),
                        vec![
                            Operand::IdRef(80),
                            Operand::IdRef(70),
                            Operand::IdRef(81),
                            Operand::IdRef(71),
                        ],
                    ),
                    Instruction::new(
                        Op::InBoundsAccessChain,
                        Some(ptr_func_float),
                        Some(83),
                        vec![Operand::IdRef(82), Operand::IdRef(idx)],
                    ),
                    Instruction::new(Op::Return, None, None, vec![]),
                ],
            },
        ],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| reroot_demoted_array_element_overindex(c, 0));

    let r = ctx.module.functions[0].blocks[2]
        .instructions
        .iter()
        .find(|i| i.result_id == Some(83))
        .expect("the over-index chain must still exist");
    assert_eq!(
        r.operands,
        vec![Operand::IdRef(arr_var), Operand::IdRef(idx)],
        "the phi-reached over-index must re-root onto the converged array variable"
    );
}

// A DYNAMIC flat word index `%buf %uint_0 (%uint_44 + %dyn)` over a typed struct binding (member 0 is
// a sub-struct, member 2 is `[4 x uint]` at byte 176 = word 44, ArrayStride 4) remaps to the typed
// dynamic array-member access `%buf %uint_2 %dyn` — byte-identical, and the chain validates.
#[test]
fn remap_dynamic_word_index_collapses_to_array_member() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let uint = 1;
    let inner = 2; // member 0: an opaque sub-struct (one uint member, irrelevant — just needs offset 0)
    let arr4 = 3; // [4 x uint]
    let outer = 4; // { inner, [4 x uint] } -- member 1 at byte 176 (word 44)
    let ptr_sb_outer = 5;
    let ptr_sb_uint = 6;
    let buf = 7;
    let uint_0 = 8;
    let uint_44 = 9;
    let uint_4 = 10;
    let dyn_id = 11; // a non-constant value (function result)
    let iadd = 12; // %iadd = uint_44 + dyn

    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(inner),
            vec![Operand::IdRef(uint)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_4),
            vec![Operand::LiteralBit32(4)],
        ),
        Instruction::new(
            Op::TypeArray,
            None,
            Some(arr4),
            vec![Operand::IdRef(uint), Operand::IdRef(uint_4)],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(outer),
            vec![Operand::IdRef(inner), Operand::IdRef(arr4)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_outer),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(outer),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_uint),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(uint),
            ],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr_sb_outer),
            Some(buf),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_0),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_44),
            vec![Operand::LiteralBit32(44)],
        ),
    ];

    // outer member offsets: member 0 @ 0, member 1 @ 176 (word 44); arr4 ArrayStride 4.
    module.annotations = vec![
        Instruction::new(
            Op::MemberDecorate,
            None,
            None,
            vec![
                Operand::IdRef(outer),
                Operand::LiteralBit32(0),
                Operand::Decoration(Decoration::Offset),
                Operand::LiteralBit32(0),
            ],
        ),
        Instruction::new(
            Op::MemberDecorate,
            None,
            None,
            vec![
                Operand::IdRef(outer),
                Operand::LiteralBit32(1),
                Operand::Decoration(Decoration::Offset),
                Operand::LiteralBit32(176),
            ],
        ),
        Instruction::new(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(arr4),
                Operand::Decoration(Decoration::ArrayStride),
                Operand::LiteralBit32(4),
            ],
        ),
    ];

    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                // %dyn is some runtime value; model it as an OpBitcast of uint_0 (non-constant result).
                Instruction::new(
                    Op::Bitcast,
                    Some(uint),
                    Some(dyn_id),
                    vec![Operand::IdRef(uint_0)],
                ),
                Instruction::new(
                    Op::IAdd,
                    Some(uint),
                    Some(iadd),
                    vec![Operand::IdRef(uint_44), Operand::IdRef(dyn_id)],
                ),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_sb_uint),
                    Some(60),
                    vec![
                        Operand::IdRef(buf),
                        Operand::IdRef(uint_0),
                        Operand::IdRef(iadd),
                    ],
                ),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| remap_dynamic_word_index_to_array_member(c, 0));

    let chain = &ctx.module.functions[0].blocks[0].instructions[2];
    assert_eq!(
        chain.operands,
        vec![
            Operand::IdRef(buf),
            // member index 1 — find the uint constant of value 1.
            chain.operands[1].clone(),
            Operand::IdRef(dyn_id),
        ],
        "chain should be %buf %uint_1 %dyn"
    );
    let Operand::IdRef(member_id) = chain.operands[1] else {
        panic!("member index not an id")
    };
    let mval = ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .find(|g| g.result_id == Some(member_id) && g.class.opcode == Op::Constant)
        .and_then(|g| match g.operands.first() {
            Some(Operand::LiteralBit32(v)) => Some(*v),
            _ => None,
        });
    assert_eq!(mval, Some(1), "word 44 should map to array member index 1");
}

// A flat dynamic word index `%buf %uint_0 (uint_12 + dyn*2)` reading a FLOAT field of a
// `[K x struct{float,float}]` member (ArrayStride 8) AS uint remaps to
// `%buf %uint_M %dyn %uint_0` (float*), and the uint load is split into `OpLoad float ; OpBitcast uint`.
#[test]
fn remap_dynamic_word_index_to_array_struct_field_remaps_and_splits_load() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let uint = 1;
    let float = 2;
    let inner = 3; // member 0: opaque sub-struct (one float, offset 0)
    let elem = 4; // ElemStruct { float, float } -- ArrayStride 8
    let arr = 5; // [2 x ElemStruct] at byte 48 (word 12)
    let outer = 6; // { inner, [2 x ElemStruct] }
    let ptr_sb_outer = 7;
    let ptr_sb_uint = 8;
    let buf = 9;
    let uint_0 = 10;
    let uint_12 = 11;
    let uint_2c = 12; // constant 2 (stride words) AND array length
    let dyn_id = 13; // non-constant value (function result)
    let imul = 14; // %imul = dyn * 2
    let iadd = 15; // %iadd = uint_12 + imul
    let chain = 16;
    let load = 17;

    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(inner),
            vec![Operand::IdRef(float)],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(elem),
            vec![Operand::IdRef(float), Operand::IdRef(float)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_2c),
            vec![Operand::LiteralBit32(2)],
        ),
        Instruction::new(
            Op::TypeArray,
            None,
            Some(arr),
            vec![Operand::IdRef(elem), Operand::IdRef(uint_2c)],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(outer),
            vec![Operand::IdRef(inner), Operand::IdRef(arr)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_outer),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(outer),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_uint),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(uint),
            ],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr_sb_outer),
            Some(buf),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_0),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_12),
            vec![Operand::LiteralBit32(12)],
        ),
    ];

    // outer: member 0 @ 0, member 1 @ 48; arr ArrayStride 8; elem fields @ 0, 4.
    module.annotations = vec![
        Instruction::new(
            Op::MemberDecorate,
            None,
            None,
            vec![
                Operand::IdRef(outer),
                Operand::LiteralBit32(0),
                Operand::Decoration(Decoration::Offset),
                Operand::LiteralBit32(0),
            ],
        ),
        Instruction::new(
            Op::MemberDecorate,
            None,
            None,
            vec![
                Operand::IdRef(outer),
                Operand::LiteralBit32(1),
                Operand::Decoration(Decoration::Offset),
                Operand::LiteralBit32(48),
            ],
        ),
        Instruction::new(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(arr),
                Operand::Decoration(Decoration::ArrayStride),
                Operand::LiteralBit32(8),
            ],
        ),
        Instruction::new(
            Op::MemberDecorate,
            None,
            None,
            vec![
                Operand::IdRef(elem),
                Operand::LiteralBit32(0),
                Operand::Decoration(Decoration::Offset),
                Operand::LiteralBit32(0),
            ],
        ),
        Instruction::new(
            Op::MemberDecorate,
            None,
            None,
            vec![
                Operand::IdRef(elem),
                Operand::LiteralBit32(1),
                Operand::Decoration(Decoration::Offset),
                Operand::LiteralBit32(4),
            ],
        ),
    ];

    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                // %dyn is a runtime value; model it as a non-constant bitcast.
                Instruction::new(
                    Op::Bitcast,
                    Some(uint),
                    Some(dyn_id),
                    vec![Operand::IdRef(uint_0)],
                ),
                Instruction::new(
                    Op::IMul,
                    Some(uint),
                    Some(imul),
                    vec![Operand::IdRef(dyn_id), Operand::IdRef(uint_2c)],
                ),
                Instruction::new(
                    Op::IAdd,
                    Some(uint),
                    Some(iadd),
                    vec![Operand::IdRef(uint_12), Operand::IdRef(imul)],
                ),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_sb_uint),
                    Some(chain),
                    vec![
                        Operand::IdRef(buf),
                        Operand::IdRef(uint_0),
                        Operand::IdRef(iadd),
                    ],
                ),
                Instruction::new(
                    Op::Load,
                    Some(uint),
                    Some(load),
                    vec![Operand::IdRef(chain)],
                ),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| {
        remap_dynamic_word_index_to_array_struct_field(c, 0)
    });

    let insts = &ctx.module.functions[0].blocks[0].instructions;
    // The access chain is retyped to a float pointer with operands %buf %uint_1 %dyn %uint_0.
    let chain_inst = insts
        .iter()
        .find(|i| i.result_id == Some(chain))
        .expect("chain");
    let Some(rt) = chain_inst.result_type else {
        panic!("chain has no result type")
    };
    let pointee = ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .find(|g| g.result_id == Some(rt) && g.class.opcode == Op::TypePointer)
        .and_then(|g| match g.operands.get(1) {
            Some(Operand::IdRef(p)) => Some(*p),
            _ => None,
        });
    assert_eq!(pointee, Some(float), "chain should be retyped to float*");
    assert_eq!(chain_inst.operands.len(), 4, "chain should be %buf M dyn F");
    assert_eq!(chain_inst.operands[0], Operand::IdRef(buf));
    assert_eq!(
        chain_inst.operands[2],
        Operand::IdRef(dyn_id),
        "element index is %dyn"
    );
    // member index = 1, field index = 0 (constants of those values).
    let const_val = |id: Operand| -> Option<u32> {
        let Operand::IdRef(cid) = id else { return None };
        ctx.new_globals
            .iter()
            .chain(ctx.module.types_global_values.iter())
            .find(|g| g.result_id == Some(cid) && g.class.opcode == Op::Constant)
            .and_then(|g| match g.operands.first() {
                Some(Operand::LiteralBit32(v)) => Some(*v),
                _ => None,
            })
    };
    assert_eq!(
        const_val(chain_inst.operands[1].clone()),
        Some(1),
        "member 1"
    );
    assert_eq!(
        const_val(chain_inst.operands[3].clone()),
        Some(0),
        "field 0"
    );

    // The original uint load id is now an OpBitcast %uint of a freshly-inserted float load.
    let bc = insts
        .iter()
        .find(|i| i.result_id == Some(load))
        .expect("load id");
    assert_eq!(bc.class.opcode, Op::Bitcast, "uint load becomes a bitcast");
    assert_eq!(
        bc.result_type,
        Some(uint),
        "bitcast preserves uint result type"
    );
    let Operand::IdRef(src) = bc.operands[0] else {
        panic!("bitcast operand")
    };
    let fload = insts
        .iter()
        .find(|i| i.result_id == Some(src))
        .expect("float load");
    assert_eq!(fload.class.opcode, Op::Load, "inserted op is a load");
    assert_eq!(
        fload.result_type,
        Some(float),
        "inserted load reads the float field"
    );
    assert_eq!(
        fload.operands[0],
        Operand::IdRef(chain),
        "float load reads the chain"
    );
}

// `OpLoad %v4float` through a `float*` access chain whose base is a `v4float*` is the emitter's
// mis-typed vector stride; the chain is rewritten to `OpPtrAccessChain %v4float* %base %idx`.
#[test]
fn repair_vector_load_through_scalar_stride_rewrites_to_ptr_access_chain() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let float = 1;
    let v4float = 2;
    let ptr_sb_v4 = 3;
    let ptr_sb_f = 4;
    let base = 5; // a v4float* value (model as a variable)
    let idx = 6;
    let chain = 7;
    let load = 8;

    module.types_global_values = vec![
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypeVector,
            None,
            Some(v4float),
            vec![Operand::IdRef(float), Operand::LiteralBit32(4)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_v4),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(v4float),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_f),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(float),
            ],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr_sb_v4),
            Some(base),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ),
        Instruction::new(
            Op::Constant,
            Some(float),
            Some(idx),
            vec![Operand::LiteralBit32(2)],
        ),
    ];

    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(float),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_sb_f),
                    Some(chain),
                    vec![Operand::IdRef(base), Operand::IdRef(idx)],
                ),
                Instruction::new(
                    Op::Load,
                    Some(v4float),
                    Some(load),
                    vec![Operand::IdRef(chain)],
                ),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| repair_vector_load_through_scalar_stride(c, 0));

    let c = &ctx.module.functions[0].blocks[0].instructions[0];
    assert_eq!(
        c.class.opcode,
        Op::PtrAccessChain,
        "chain becomes PtrAccessChain"
    );
    assert_eq!(c.result_type, Some(ptr_sb_v4), "retyped to v4float*");
    assert_eq!(
        c.operands,
        vec![Operand::IdRef(base), Operand::IdRef(idx)],
        "operands unchanged (base + idx)"
    );
}

// `OpLoad %float` through a `v4float*` PtrAccessChain is the matrix-column gather's strided component
// read; the chain gets a trailing `%uint_0` and is retyped to `float*`.
#[test]
fn repair_scalar_load_through_vector_ptr_appends_component_zero() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let uint = 1;
    let float = 2;
    let v4float = 3;
    let ptr_sb_v4 = 4;
    let ptr_sb_f = 5;
    let base = 6;
    let uint_1 = 7;
    let chain = 8;
    let load = 9;

    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypeVector,
            None,
            Some(v4float),
            vec![Operand::IdRef(float), Operand::LiteralBit32(4)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_v4),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(v4float),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_f),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(float),
            ],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr_sb_v4),
            Some(base),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_1),
            vec![Operand::LiteralBit32(1)],
        ),
    ];

    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(float),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::PtrAccessChain,
                    Some(ptr_sb_v4),
                    Some(chain),
                    vec![Operand::IdRef(base), Operand::IdRef(uint_1)],
                ),
                Instruction::new(
                    Op::Load,
                    Some(float),
                    Some(load),
                    vec![Operand::IdRef(chain)],
                ),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| repair_scalar_load_through_vector_ptr(c, 0));

    let c = &ctx.module.functions[0].blocks[0].instructions[0];
    assert_eq!(c.result_type, Some(ptr_sb_f), "retyped to float*");
    assert_eq!(
        c.operands.len(),
        3,
        "a trailing component index was appended"
    );
    let Operand::IdRef(last) = c.operands[2] else {
        panic!("trailing index not an id")
    };
    let lval = ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .find(|g| g.result_id == Some(last) && g.class.opcode == Op::Constant)
        .and_then(|g| match g.operands.first() {
            Some(Operand::LiteralBit32(v)) => Some(*v),
            _ => None,
        });
    assert_eq!(lval, Some(0), "appended component index is 0");
}

// A write-only `Function` ulong array with a type-mismatched store (a StorageBuffer pointer stored into
// a ulong slot) is recognized as dead; its stores and access chains are removed.
#[test]
fn drop_writeonly_dead_local_array_stores_removes_invalid_stores() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let ulong = 1;
    let uint = 2;
    let float = 3;
    let arr16 = 4;
    let ptr_fn_arr = 5;
    let ptr_fn_ulong = 6;
    let ptr_sb_float = 7;
    let arr_var = 8;
    let buf = 9; // a StorageBuffer float* value to store (type-mismatched into ulong slot)
    let uint_16 = 10;
    let idx0 = 11;
    let slot = 12;

    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(ulong),
            vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(uint_16),
            vec![Operand::LiteralBit32(16)],
        ),
        Instruction::new(
            Op::TypeArray,
            None,
            Some(arr16),
            vec![Operand::IdRef(ulong), Operand::IdRef(uint_16)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_fn_arr),
            vec![
                Operand::StorageClass(StorageClass::Function),
                Operand::IdRef(arr16),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_fn_ulong),
            vec![
                Operand::StorageClass(StorageClass::Function),
                Operand::IdRef(ulong),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_float),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(float),
            ],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr_sb_float),
            Some(buf),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(idx0),
            vec![Operand::LiteralBit32(0)],
        ),
    ];

    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            None,
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::Variable,
                    Some(ptr_fn_arr),
                    Some(arr_var),
                    vec![Operand::StorageClass(StorageClass::Function)],
                ),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_fn_ulong),
                    Some(slot),
                    vec![Operand::IdRef(arr_var), Operand::IdRef(idx0)],
                ),
                // store a StorageBuffer pointer into the ulong slot — the invalid residue.
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(slot), Operand::IdRef(buf)],
                ),
                Instruction::new(Op::Return, None, None, vec![]),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| drop_writeonly_dead_local_array_stores(c, 0));

    let body = &ctx.module.functions[0].blocks[0].instructions;
    assert!(
        !body.iter().any(|i| i.class.opcode == Op::Store),
        "the dead invalid store should be removed"
    );
    assert!(
        !body.iter().any(|i| i.result_id == Some(slot)),
        "the now-dead access chain should be removed"
    );
    assert!(
        body.iter().any(|i| i.result_id == Some(arr_var)),
        "the (now unused) Function variable may remain"
    );
}

// --- Additional coverage for previously-untested access.rs repair passes (refactor T6/§6) --------
//
// These four passes ran in the `transform_with_options` "2b/2c/2d/2h" blocks with zero isolated unit
// coverage (§6 gap-list). Each fixture is a hand-built crate module exercising the pass's transform
// on a minimal shape, asserting the concrete rewrite. Three are genuinely idempotent (their driver
// slots assume it), so they run through `run_idempotent`; `guard_integer_division_by_zero` is NOT
// idempotent by design (a second run re-guards the already-guarded denominator) — the driver runs it
// exactly once, so it is tested with a single call and the non-idempotence is documented, not fixed.

/// A same-width int/float `OpUConvert`/`OpSConvert`/`OpFConvert` is a no-op the SPIR-V would reject as
/// a width-preserving convert; `fix_noop_width_converts` rewrites it to `OpCopyObject`. A genuine
/// narrowing/widening convert (differing scalar bit width) is left untouched.
#[test]
fn fix_noop_width_converts_rewrites_samewidth_convert_to_copyobject() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let uint = 1; // TypeInt 32 0
    let ushort = 2; // TypeInt 16 0
    let p_uint = 10; // param : uint  (same width as the convert result)
    let p_ushort = 11; // param : ushort (genuinely narrower source)
    let noop_res = 20;
    let real_res = 21;
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(ushort),
            vec![Operand::LiteralBit32(16), Operand::LiteralBit32(0)],
        ),
    ];
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![
            Instruction::new(Op::FunctionParameter, Some(uint), Some(p_uint), vec![]),
            Instruction::new(Op::FunctionParameter, Some(ushort), Some(p_ushort), vec![]),
        ],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                // Same width (uint -> uint): the convert is a no-op.
                Instruction::new(
                    Op::UConvert,
                    Some(uint),
                    Some(noop_res),
                    vec![Operand::IdRef(p_uint)],
                ),
                // Real widening (ushort -> uint): must stay a convert.
                Instruction::new(
                    Op::UConvert,
                    Some(uint),
                    Some(real_res),
                    vec![Operand::IdRef(p_ushort)],
                ),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| fix_noop_width_converts(c, 0));

    let body = &ctx.module.functions[0].blocks[0].instructions;
    assert_eq!(
        body[0].class.opcode,
        Op::CopyObject,
        "same-width convert should become OpCopyObject"
    );
    assert_eq!(body[0].result_id, Some(noop_res));
    assert_eq!(body[0].result_type, Some(uint));
    assert_eq!(body[0].operands, vec![Operand::IdRef(p_uint)]);
    assert_eq!(
        body[1].class.opcode,
        Op::UConvert,
        "a genuine width-changing convert must be left untouched"
    );
    assert_eq!(body[1].result_id, Some(real_res));
}

/// `fix_merge_placement` slides an `OpSelectionMerge`/`OpLoopMerge` that llc's structurizer left
/// mid-block down to sit immediately before the block's branch terminator (SPIR-V requires the merge
/// to be the second-to-last instruction), preserving the relative order of the displaced values.
#[test]
fn fix_merge_placement_slides_midblock_merge_before_terminator() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let uint = 1;
    let boolt = 2;
    let cond = 10; // param : bool
    let merge_label = 40;
    let target_a = 41;
    let target_b = 42;
    let hoisted = 30;
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(Op::TypeBool, None, Some(boolt), vec![]),
    ];
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![Instruction::new(
            Op::FunctionParameter,
            Some(boolt),
            Some(cond),
            vec![],
        )],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                // Merge left MID-BLOCK (index 0), separated from the branch by a hoisted value.
                Instruction::new(
                    Op::SelectionMerge,
                    None,
                    None,
                    vec![
                        Operand::IdRef(merge_label),
                        Operand::SelectionControl(SelectionControl::NONE),
                    ],
                ),
                Instruction::new(Op::Undef, Some(uint), Some(hoisted), vec![]),
                Instruction::new(
                    Op::BranchConditional,
                    None,
                    None,
                    vec![
                        Operand::IdRef(cond),
                        Operand::IdRef(target_a),
                        Operand::IdRef(target_b),
                    ],
                ),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| fix_merge_placement(c, 0));

    let body = &ctx.module.functions[0].blocks[0].instructions;
    let n = body.len();
    assert_eq!(n, 3, "no instructions added or removed, only reordered");
    assert_eq!(
        body[n - 2].class.opcode,
        Op::SelectionMerge,
        "the merge must sit immediately before the terminator"
    );
    assert_eq!(
        body[n - 1].class.opcode,
        Op::BranchConditional,
        "the branch stays last"
    );
    assert_eq!(
        body[n - 3].result_id,
        Some(hoisted),
        "the displaced value keeps its relative order above the merge"
    );
}

/// `normalize_int_arith_operand_widths` inserts a truncating `OpUConvert` for any integer-arithmetic
/// operand wider than the instruction's result type (spirv-val rejects a mismatched-width operand),
/// deduplicating within one instruction so a value used twice truncates once.
#[test]
fn normalize_int_arith_operand_widths_truncates_wider_operand() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let uint = 1; // TypeInt 32 0
    let ulong = 2; // TypeInt 64 0
    let p_ulong = 10; // param : ulong, reused as both IMul operands
    let mul_res = 20;
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(ulong),
            vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
        ),
    ];
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![Instruction::new(
            Op::FunctionParameter,
            Some(ulong),
            Some(p_ulong),
            vec![],
        )],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            // 32-bit result, 64-bit operands (twice) -> one truncating UConvert, both operands rebound.
            instructions: vec![Instruction::new(
                Op::IMul,
                Some(uint),
                Some(mul_res),
                vec![Operand::IdRef(p_ulong), Operand::IdRef(p_ulong)],
            )],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, normalize_int_arith_operand_widths);

    let body = &ctx.module.functions[0].blocks[0].instructions;
    assert_eq!(
        body.len(),
        2,
        "one truncating UConvert inserted before the IMul"
    );
    assert_eq!(
        body[0].class.opcode,
        Op::UConvert,
        "the wide operand is truncated to the result width"
    );
    assert_eq!(body[0].result_type, Some(uint));
    assert_eq!(body[0].operands, vec![Operand::IdRef(p_ulong)]);
    let narrow = body[0].result_id.expect("truncation has a result id");
    assert_eq!(body[1].class.opcode, Op::IMul);
    assert_eq!(
        body[1].operands,
        vec![Operand::IdRef(narrow), Operand::IdRef(narrow)],
        "both IMul operands rebind to the single deduplicated truncation"
    );
}

/// `guard_integer_division_by_zero` guards an eager `OpUDiv`/`OpSDiv`/`OpUMod`/`OpSRem` denominator
/// so the otherwise-undefined divide-by-zero arm becomes deterministic: it inserts `denom == 0` and a
/// select of `1` for the zero case, rebinding the divide's denominator to the guarded value.
///
/// NOTE: this pass is deliberately NOT idempotent — a second run would re-guard the already-guarded
/// (select) denominator. The driver runs it exactly once (slot "2c"), so the transform is asserted
/// with a single call; the non-idempotence is a documented property, not a defect to paper over.
#[test]
fn guard_integer_division_by_zero_inserts_denominator_guard() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let uint = 1;
    let numer = 10; // param : uint
    let denom = 11; // param : uint (the guarded denominator)
    let div_res = 20;
    module.types_global_values = vec![Instruction::new(
        Op::TypeInt,
        None,
        Some(uint),
        vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
    )];
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![
            Instruction::new(Op::FunctionParameter, Some(uint), Some(numer), vec![]),
            Instruction::new(Op::FunctionParameter, Some(uint), Some(denom), vec![]),
        ],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![Instruction::new(
                Op::UDiv,
                Some(uint),
                Some(div_res),
                vec![Operand::IdRef(numer), Operand::IdRef(denom)],
            )],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    // Single call: this pass is not idempotent (see doc comment).
    guard_integer_division_by_zero(&mut ctx, 0);

    let body = &ctx.module.functions[0].blocks[0].instructions;
    assert_eq!(
        body.len(),
        3,
        "an IEqual + Select guard is inserted before the divide"
    );
    assert_eq!(body[0].class.opcode, Op::IEqual, "denominator == 0 test");
    assert_eq!(
        body[0].operands[0],
        Operand::IdRef(denom),
        "the zero test reads the original denominator"
    );
    assert_eq!(body[1].class.opcode, Op::Select, "select 1 when denom == 0");
    let safe = body[1].result_id.expect("select has a result id");
    let div = &body[2];
    assert_eq!(div.class.opcode, Op::UDiv, "the divide itself is preserved");
    assert_eq!(div.result_id, Some(div_res));
    assert_eq!(
        div.operands[0],
        Operand::IdRef(numer),
        "numerator unchanged"
    );
    assert_eq!(
        div.operands[1],
        Operand::IdRef(safe),
        "the divide denominator rebinds to the guarded (select) value"
    );
}

/// `narrow_access_chain_indices` rewrites a CONSTANT 64-bit access-chain index (which NVIDIA's
/// SPIR-V->NVVM compiler crashes on) to the equal-valued 32-bit `uint` constant when it fits in
/// u32; a dynamic 64-bit index (a runtime value) and an already-32-bit index are both left alone.
#[test]
fn narrow_access_chain_indices_narrows_constant_i64_index() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let ulong = 1; // TypeInt 64 0
    let uint = 2; // TypeInt 32 0
    let ptr = 3; // _ptr_StorageBuffer_uint (base + result; not inspected by the pass)
    let base = 10; // Variable (base pointer)
    let c64 = 11; // Constant ulong 5 (the 64-bit index to narrow)
    let c32 = 12; // Constant uint 3 (an already-32-bit index, must stay)
    let chain_a = 20;
    let chain_b = 21;
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(ulong),
            vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(uint),
            ],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr),
            Some(base),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ),
        Instruction::new(
            Op::Constant,
            Some(ulong),
            Some(c64),
            vec![Operand::LiteralBit64(5)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(c32),
            vec![Operand::LiteralBit32(3)],
        ),
    ];
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr),
                    Some(chain_a),
                    vec![Operand::IdRef(base), Operand::IdRef(c64)],
                ),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr),
                    Some(chain_b),
                    vec![Operand::IdRef(base), Operand::IdRef(c32)],
                ),
            ],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| narrow_access_chain_indices(c, 0));

    let body = &ctx.module.functions[0].blocks[0].instructions;
    // Chain A's 64-bit index is replaced with a 32-bit uint constant of the same value (5).
    let Some(Operand::IdRef(new_idx)) = body[0].operands.get(1) else {
        panic!("chain A lost its index operand");
    };
    assert_ne!(*new_idx, c64, "the 64-bit constant index must be replaced");
    let narrowed = ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .find(|g| g.result_id == Some(*new_idx))
        .expect("the narrowed index constant must exist");
    assert_eq!(narrowed.class.opcode, Op::Constant);
    assert_eq!(
        narrowed.result_type,
        Some(uint),
        "the narrowed index is a 32-bit uint constant"
    );
    assert_eq!(narrowed.operands, vec![Operand::LiteralBit32(5)]);
    // Chain B's already-32-bit index is untouched.
    assert_eq!(
        body[1].operands.get(1),
        Some(&Operand::IdRef(c32)),
        "an already-32-bit index must be left alone"
    );
}

/// `drop_overindexed_zero_tail` truncates a trailing run of constant-zero over-indices from an
/// otherwise-INVALID member-access chain: when the surviving prefix already reaches the declared
/// scalar pointee, the leftover member-0 (byte-offset-0) descents are a byte no-op, so dropping them
/// yields the valid chain at the identical address. A valid chain (walk consumes every index) is
/// untouched.
#[test]
fn drop_overindexed_zero_tail_truncates_trailing_zero_overindex() {
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));

    let uint = 1; // TypeInt 32 0 (the flattened scalar)
    let struct_s = 2; // TypeStruct { uint } (member 0 is the scalar)
    let ptr_struct = 3; // _ptr_StorageBuffer_S (base pointer)
    let ptr_uint = 4; // _ptr_StorageBuffer_uint (chain result: the scalar)
    let base = 10; // Variable (base)
    let c0 = 11; // Constant uint 0 (member-0 index, reused for the over-index tail)
    let chain = 20;
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_s),
            vec![Operand::IdRef(uint)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_struct),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(struct_s),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_uint),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(uint),
            ],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr_struct),
            Some(base),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(c0),
            vec![Operand::LiteralBit32(0)],
        ),
    ];
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(50),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(51),
            ],
        )),
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            // `base [0, 0]`: index 0 descends into member 0 (uint); the second 0 OVER-indexes the
            // reached scalar -> currently invalid, and the trailing zero is a byte no-op to drop.
            instructions: vec![Instruction::new(
                Op::InBoundsAccessChain,
                Some(ptr_uint),
                Some(chain),
                vec![Operand::IdRef(base), Operand::IdRef(c0), Operand::IdRef(c0)],
            )],
        }],
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| drop_overindexed_zero_tail(c, 0));

    let chain_inst = &ctx.module.functions[0].blocks[0].instructions[0];
    assert_eq!(
        chain_inst.operands,
        vec![Operand::IdRef(base), Operand::IdRef(c0)],
        "the trailing zero over-index is dropped, leaving the valid single-index chain"
    );
}

// The pass is byte-neutral where it must not fire: an arithmetic op whose operands already match the
// result width is left alone, and a shift is excluded entirely (its Shift operand may legally differ
// in width from the result), so no `OpUConvert` is inserted for either.
#[test]
fn normalize_int_arith_operand_widths_leaves_matched_and_shift_operands_alone() {
    let uint = 1;
    let ulong = 2;
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(ulong),
            vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
        ),
    ];

    let wide = 50;
    let narrow = 51;
    let add = 60;
    let shift = 61;
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(40),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(41),
            ],
        )),
        parameters: vec![
            Instruction::new(Op::FunctionParameter, Some(ulong), Some(wide), vec![]),
            Instruction::new(Op::FunctionParameter, Some(uint), Some(narrow), vec![]),
        ],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                // Matched widths (uint + uint): no coercion needed.
                Instruction::new(
                    Op::IAdd,
                    Some(uint),
                    Some(add),
                    vec![Operand::IdRef(narrow), Operand::IdRef(narrow)],
                ),
                // A shift with a WIDER (ulong) shift operand: excluded, so left as-is.
                Instruction::new(
                    Op::ShiftLeftLogical,
                    Some(uint),
                    Some(shift),
                    vec![Operand::IdRef(narrow), Operand::IdRef(wide)],
                ),
            ],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, normalize_int_arith_operand_widths);

    let body = &ctx.module.functions[0].blocks[0].instructions;
    assert!(
        !body.iter().any(|i| i.class.opcode == Op::UConvert),
        "no truncation is inserted for matched-width or shift operands"
    );
    let add_inst = body.iter().find(|i| i.result_id == Some(add)).unwrap();
    assert_eq!(
        add_inst.operands,
        vec![Operand::IdRef(narrow), Operand::IdRef(narrow)],
        "the matched-width add is untouched"
    );
    let shift_inst = body.iter().find(|i| i.result_id == Some(shift)).unwrap();
    assert_eq!(
        shift_inst.operands,
        vec![Operand::IdRef(narrow), Operand::IdRef(wide)],
        "the shift keeps its wider shift operand"
    );
}

// A wider-scalar `OpStore` through a NARROWER element pointer (a `uint` object written through a
// `device uchar*` `OpPtrAccessChain` — the pointee-mismatch spirv-val rejects) is lowered to
// `obj_bits / pointee_bits` little-endian per-element stores through sibling element pointers.
#[test]
fn lower_subword_scalar_store_splits_wide_store_into_per_element_little_endian_stores() {
    let uchar = 1;
    let uint = 2;
    let ptr_sb_uchar = 3;
    let idx0 = 4;
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uchar),
            vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_uchar),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(uchar),
            ],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(idx0),
            vec![Operand::LiteralBit32(0)],
        ),
    ];

    // `%elem = OpPtrAccessChain %uchar* %base 0` is a genuine byte element pointer; storing the 32-bit
    // `%obj` through it mismatches the `uchar` pointee — the shape the pass lowers.
    let base = 50;
    let obj = 51;
    let elem = 60;
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(40),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(41),
            ],
        )),
        parameters: vec![
            Instruction::new(
                Op::FunctionParameter,
                Some(ptr_sb_uchar),
                Some(base),
                vec![],
            ),
            Instruction::new(Op::FunctionParameter, Some(uint), Some(obj), vec![]),
        ],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::PtrAccessChain,
                    Some(ptr_sb_uchar),
                    Some(elem),
                    vec![Operand::IdRef(base), Operand::IdRef(idx0)],
                ),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(elem), Operand::IdRef(obj)],
                ),
            ],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| lower_subword_scalar_store(c, 0));

    let body = &ctx.module.functions[0].blocks[0].instructions;
    // 32-bit object / 8-bit pointee = 4 little-endian slot stores; the single wide store is gone.
    let stores: Vec<&Instruction> = body
        .iter()
        .filter(|i| i.class.opcode == Op::Store)
        .collect();
    assert_eq!(
        stores.len(),
        4,
        "one store per 8-bit slot of the 32-bit object"
    );
    assert!(
        !stores
            .iter()
            .any(|s| s.operands.get(1) == Some(&Operand::IdRef(obj))),
        "the wide object is never stored directly through the byte pointer"
    );
    // Each slot value is truncated to the pointee width; slots 1..4 shift the object down first.
    let truncations = body
        .iter()
        .filter(|i| i.class.opcode == Op::UConvert && i.result_type == Some(uchar))
        .count();
    assert_eq!(
        truncations, 4,
        "each slot narrows to the uchar pointee width"
    );
    let shifts = body
        .iter()
        .filter(|i| i.class.opcode == Op::ShiftRightLogical)
        .count();
    assert_eq!(
        shifts, 3,
        "slots 1,2,3 shift the object right; slot 0 does not"
    );
    // 1 base chain (defining %elem) + 3 sibling element pointers for slots 1,2,3.
    let chains = body
        .iter()
        .filter(|i| i.class.opcode == Op::PtrAccessChain)
        .count();
    assert_eq!(
        chains, 4,
        "three sibling element pointers plus the original base chain"
    );
}

#[test]
fn lower_subword_scalar_store_splits_wide_store_through_array_element_access_chain() {
    let uchar = 1;
    let uint = 2;
    let runtime_uchar = 3;
    let block_ty = 4;
    let ptr_sb_block = 5;
    let ptr_sb_uchar = 6;
    let member0 = 7;
    let dyn_idx = 8;
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uchar),
            vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeRuntimeArray,
            None,
            Some(runtime_uchar),
            vec![Operand::IdRef(uchar)],
        ),
        Instruction::new(
            Op::TypeStruct,
            None,
            Some(block_ty),
            vec![Operand::IdRef(runtime_uchar)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_block),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(block_ty),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_uchar),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(uchar),
            ],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(member0),
            vec![Operand::LiteralBit32(0)],
        ),
    ];

    let base = 50;
    let obj = 51;
    let elem = 60;
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(40),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(41),
            ],
        )),
        parameters: vec![
            Instruction::new(
                Op::FunctionParameter,
                Some(ptr_sb_block),
                Some(base),
                vec![],
            ),
            Instruction::new(Op::FunctionParameter, Some(uint), Some(obj), vec![]),
            Instruction::new(Op::FunctionParameter, Some(uint), Some(dyn_idx), vec![]),
        ],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_sb_uchar),
                    Some(elem),
                    vec![
                        Operand::IdRef(base),
                        Operand::IdRef(member0),
                        Operand::IdRef(dyn_idx),
                    ],
                ),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(elem), Operand::IdRef(obj)],
                ),
            ],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| lower_subword_scalar_store(c, 0));

    let body = &ctx.module.functions[0].blocks[0].instructions;
    assert_eq!(
        body.iter().filter(|i| i.class.opcode == Op::Store).count(),
        4,
        "the wide store is split even when the element pointer is a composed access-chain"
    );
    assert!(
        !body.iter().any(|i| i.class.opcode == Op::Store
            && i.operands.get(1) == Some(&Operand::IdRef(obj))),
        "the original wide object is not stored directly through the uchar pointer"
    );
    assert_eq!(
        body.iter()
            .filter(|i| i.class.opcode == Op::PtrAccessChain)
            .count(),
        3,
        "slots 1..3 use sibling element pointers from the access-chain element"
    );
}

#[test]
fn lower_subword_scalar_store_splits_vector_into_scalar_element_stores() {
    let ushort = 1;
    let ulong = 2;
    let v4ushort = 3;
    let ptr_sb_ushort = 4;
    let idx0 = 5;
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(ushort),
            vec![Operand::LiteralBit32(16), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(ulong),
            vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeVector,
            None,
            Some(v4ushort),
            vec![Operand::IdRef(ushort), Operand::LiteralBit32(4)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_ushort),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(ushort),
            ],
        ),
        Instruction::new(
            Op::Constant,
            Some(ulong),
            Some(idx0),
            vec![Operand::LiteralBit64(0)],
        ),
    ];

    let base = 50;
    let object = 51;
    let element = 60;
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(ulong),
            Some(40),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(41),
            ],
        )),
        parameters: vec![
            Instruction::new(
                Op::FunctionParameter,
                Some(ptr_sb_ushort),
                Some(base),
                vec![],
            ),
            Instruction::new(Op::FunctionParameter, Some(v4ushort), Some(object), vec![]),
        ],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::PtrAccessChain,
                    Some(ptr_sb_ushort),
                    Some(element),
                    vec![Operand::IdRef(base), Operand::IdRef(idx0)],
                ),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(element), Operand::IdRef(object)],
                ),
            ],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| lower_subword_scalar_store(c, 0));

    let body = &ctx.module.functions[0].blocks[0].instructions;
    assert_eq!(
        body.iter()
            .filter(|inst| inst.class.opcode == Op::Store)
            .count(),
        4,
        "one scalar store per vector lane"
    );
    assert!(
        !body.iter().any(|inst| inst.class.opcode == Op::Store
            && inst.operands.get(1) == Some(&Operand::IdRef(object))),
        "the mismatched vector store is removed"
    );
    assert!(
        body.iter()
            .any(|inst| inst.class.opcode == Op::Bitcast && inst.result_type == Some(ulong)),
        "the vector payload is reinterpreted once as its 64-bit bit pattern"
    );
}

// Floor-safety: the pass never fires where a valid module already matches. A same-width store through
// the byte element pointer (matched pointee) is untouched, and a wider store whose pointer is NOT an
// `OpPtrAccessChain` element pointer (here a bare parameter) is left alone.
#[test]
fn lower_subword_scalar_store_leaves_matched_and_non_element_stores_alone() {
    let uchar = 1;
    let uint = 2;
    let ptr_sb_uchar = 3;
    let ptr_sb_uint = 4;
    let idx0 = 5;
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uchar),
            vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_uchar),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(uchar),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_uint),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(uint),
            ],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(idx0),
            vec![Operand::LiteralBit32(0)],
        ),
    ];

    let base = 50;
    let byte_obj = 51; // a uchar value — matches the uchar pointee
    let wide_obj = 52; // a uint value stored through a bare param pointer (not an element chain)
    let wide_param_ptr = 53; // a `uint*` parameter used directly as a store pointer
    let elem = 60;
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(uint),
            Some(40),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(41),
            ],
        )),
        parameters: vec![
            Instruction::new(
                Op::FunctionParameter,
                Some(ptr_sb_uchar),
                Some(base),
                vec![],
            ),
            Instruction::new(Op::FunctionParameter, Some(uchar), Some(byte_obj), vec![]),
            Instruction::new(Op::FunctionParameter, Some(uint), Some(wide_obj), vec![]),
            Instruction::new(
                Op::FunctionParameter,
                Some(ptr_sb_uint),
                Some(wide_param_ptr),
                vec![],
            ),
        ],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::PtrAccessChain,
                    Some(ptr_sb_uchar),
                    Some(elem),
                    vec![Operand::IdRef(base), Operand::IdRef(idx0)],
                ),
                // Matched: uchar object through the uchar element pointer.
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(elem), Operand::IdRef(byte_obj)],
                ),
                // Wide object, but the pointer is a bare parameter (no PtrAccessChain def): not an
                // element pointer, so the pass declines it.
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(wide_param_ptr), Operand::IdRef(wide_obj)],
                ),
            ],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| lower_subword_scalar_store(c, 0));

    let body = &ctx.module.functions[0].blocks[0].instructions;
    // Both original stores survive verbatim; no slotting occurred.
    assert_eq!(
        body.iter().filter(|i| i.class.opcode == Op::Store).count(),
        2,
        "neither store is split"
    );
    assert!(
        body.iter().any(|i| i.class.opcode == Op::Store
            && i.operands == vec![Operand::IdRef(elem), Operand::IdRef(byte_obj)]),
        "the matched byte store is untouched"
    );
    assert!(
        body.iter().any(|i| i.class.opcode == Op::Store
            && i.operands == vec![Operand::IdRef(wide_param_ptr), Operand::IdRef(wide_obj)]),
        "the wide store through a non-element pointer is untouched"
    );
    assert!(
        !body.iter().any(|i| i.class.opcode == Op::ShiftRightLogical),
        "no slotting arithmetic is emitted"
    );
}

// An access chain rooted at a null pointer constant (an `OpConstantNull` of pointer type) cannot be
// dereferenced in Logical SPIR-V, so the pass poisons the whole null-derived chain: the chain becomes
// an `OpUndef` of its pointer result type (and its result is itself tracked as null), a load through a
// null pointer becomes an `OpCopyObject` of a fresh null constant, and a store through one is dropped.
#[test]
fn neutralize_null_access_chains_poisons_null_derived_chain_load_and_store() {
    let float = 1;
    let ptr_sb_float = 2;
    let uint = 3;
    let idx0 = 4;
    let null_base = 5;
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_float),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(float),
            ],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(idx0),
            vec![Operand::LiteralBit32(0)],
        ),
        // A null pointer constant — the poisoned root.
        Instruction::new(
            Op::ConstantNull,
            Some(ptr_sb_float),
            Some(null_base),
            vec![],
        ),
    ];

    let chain = 60;
    let val = 61;
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(float),
            Some(40),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(41),
            ],
        )),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::AccessChain,
                    Some(ptr_sb_float),
                    Some(chain),
                    vec![Operand::IdRef(null_base), Operand::IdRef(idx0)],
                ),
                Instruction::new(
                    Op::Load,
                    Some(float),
                    Some(val),
                    vec![Operand::IdRef(chain)],
                ),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(null_base), Operand::IdRef(val)],
                ),
            ],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| neutralize_null_access_chains(c, 0));

    let body = &ctx.module.functions[0].blocks[0].instructions;
    assert!(
        !body.iter().any(|i| matches!(
            i.class.opcode,
            Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
        )),
        "the null-rooted chain is removed"
    );
    assert!(
        !body.iter().any(|i| i.class.opcode == Op::Load),
        "the load through the null pointer is removed"
    );
    assert!(
        !body.iter().any(|i| i.class.opcode == Op::Store),
        "the store through the null pointer is dropped entirely"
    );
    // The chain's pointer result becomes an OpUndef of the same pointer type, keeping its id.
    let undef = body
        .iter()
        .find(|i| i.result_id == Some(chain))
        .expect("the chain result id survives as a poisoned value");
    assert_eq!(undef.class.opcode, Op::Undef);
    assert_eq!(undef.result_type, Some(ptr_sb_float));
    // The load's scalar result becomes an OpCopyObject of a fresh null float constant.
    let copy = body
        .iter()
        .find(|i| i.result_id == Some(val))
        .expect("the load result id survives as a null copy");
    assert_eq!(copy.class.opcode, Op::CopyObject);
    assert_eq!(copy.result_type, Some(float));
    let Some(Operand::IdRef(zero)) = copy.operands.first() else {
        panic!("copy sources a value id");
    };
    let zero_def = ctx
        .new_globals
        .iter()
        .find(|i| i.result_id == Some(*zero))
        .expect("the null source is a synthesized global");
    assert_eq!(zero_def.class.opcode, Op::ConstantNull);
    assert_eq!(zero_def.result_type, Some(float));
}

// The pass only fires on chains/loads/stores rooted at a null-pointer constant: an ordinary chain over
// a live pointer, and its load/store, are left byte-for-byte alone.
#[test]
fn neutralize_null_access_chains_leaves_live_pointer_access_alone() {
    let float = 1;
    let ptr_sb_float = 2;
    let uint = 3;
    let idx0 = 4;
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_float),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(float),
            ],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(idx0),
            vec![Operand::LiteralBit32(0)],
        ),
    ];

    // `%base` is a live (non-null) pointer parameter.
    let base = 50;
    let chain = 60;
    let val = 61;
    let chain_ops = vec![Operand::IdRef(base), Operand::IdRef(idx0)];
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(float),
            Some(40),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(41),
            ],
        )),
        parameters: vec![Instruction::new(
            Op::FunctionParameter,
            Some(ptr_sb_float),
            Some(base),
            vec![],
        )],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::AccessChain,
                    Some(ptr_sb_float),
                    Some(chain),
                    chain_ops.clone(),
                ),
                Instruction::new(
                    Op::Load,
                    Some(float),
                    Some(val),
                    vec![Operand::IdRef(chain)],
                ),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(chain), Operand::IdRef(val)],
                ),
            ],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| neutralize_null_access_chains(c, 0));

    let body = &ctx.module.functions[0].blocks[0].instructions;
    assert_eq!(body.len(), 3, "no instruction is added or removed");
    let chain_inst = body.iter().find(|i| i.result_id == Some(chain)).unwrap();
    assert_eq!(chain_inst.class.opcode, Op::AccessChain);
    assert_eq!(chain_inst.operands, chain_ops);
    assert!(
        body.iter()
            .any(|i| i.class.opcode == Op::Load && i.result_id == Some(val)),
        "the load over the live pointer survives"
    );
    assert!(
        body.iter().any(|i| i.class.opcode == Op::Store),
        "the store over the live pointer survives"
    );
}

// A chain rooted at an unnamed Private placeholder variable (a null-initialized `Private` `OpVariable`
// with no debug name — a demoted inline aggregate the emitter cannot address) is replaced by an
// `OpCopyObject` of a freshly synthesized zero private pointer, and the chain result is itself tracked
// as a placeholder root.
#[test]
fn neutralize_private_placeholder_access_chains_replaces_unnamed_null_private_chain() {
    let float = 1;
    let uint = 2;
    let len2 = 3;
    let arr = 4;
    let ptr_priv_arr = 5;
    let ptr_priv_float = 6;
    let null_arr = 7;
    let idx0 = 8;
    let var = 9;
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(len2),
            vec![Operand::LiteralBit32(2)],
        ),
        Instruction::new(
            Op::TypeArray,
            None,
            Some(arr),
            vec![Operand::IdRef(float), Operand::IdRef(len2)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_priv_arr),
            vec![
                Operand::StorageClass(StorageClass::Private),
                Operand::IdRef(arr),
            ],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_priv_float),
            vec![
                Operand::StorageClass(StorageClass::Private),
                Operand::IdRef(float),
            ],
        ),
        Instruction::new(Op::ConstantNull, Some(arr), Some(null_arr), vec![]),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(idx0),
            vec![Operand::LiteralBit32(0)],
        ),
        // Unnamed Private variable initialized to null — the placeholder root.
        Instruction::new(
            Op::Variable,
            Some(ptr_priv_arr),
            Some(var),
            vec![
                Operand::StorageClass(StorageClass::Private),
                Operand::IdRef(null_arr),
            ],
        ),
    ];

    let chain = 60;
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(float),
            Some(40),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(41),
            ],
        )),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![Instruction::new(
                Op::AccessChain,
                Some(ptr_priv_float),
                Some(chain),
                vec![Operand::IdRef(var), Operand::IdRef(idx0)],
            )],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| {
        neutralize_private_placeholder_access_chains(c, 0).unwrap()
    });

    let body = &ctx.module.functions[0].blocks[0].instructions;
    assert!(
        !body.iter().any(|i| i.class.opcode == Op::AccessChain),
        "the placeholder-rooted chain is removed"
    );
    let copy = body
        .iter()
        .find(|i| i.result_id == Some(chain))
        .expect("the chain result id survives as a copy");
    assert_eq!(copy.class.opcode, Op::CopyObject);
    assert_eq!(copy.result_type, Some(ptr_priv_float));
    let Some(Operand::IdRef(placeholder)) = copy.operands.first() else {
        panic!("copy sources a placeholder pointer id");
    };
    // The placeholder is a synthesized null-initialized Private variable of the chain's pointer type.
    let ph_def = ctx
        .new_globals
        .iter()
        .find(|i| i.result_id == Some(*placeholder))
        .expect("placeholder pointer is a synthesized global");
    assert_eq!(ph_def.class.opcode, Op::Variable);
    assert_eq!(ph_def.result_type, Some(ptr_priv_float));
    assert_eq!(
        ph_def.operands.first(),
        Some(&Operand::StorageClass(StorageClass::Private))
    );
}

// The pass must NOT touch a Private variable that carries a debug name (a real source-level global, not
// an emitter placeholder): its access chain is left byte-for-byte alone.
#[test]
fn neutralize_private_placeholder_access_chains_spares_named_private_variable() {
    let float = 1;
    let uint = 2;
    let ptr_priv_float = 3;
    let null_float = 4;
    let idx0 = 5;
    let var = 6;
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_priv_float),
            vec![
                Operand::StorageClass(StorageClass::Private),
                Operand::IdRef(float),
            ],
        ),
        Instruction::new(Op::ConstantNull, Some(float), Some(null_float), vec![]),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(idx0),
            vec![Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Variable,
            Some(ptr_priv_float),
            Some(var),
            vec![
                Operand::StorageClass(StorageClass::Private),
                Operand::IdRef(null_float),
            ],
        ),
    ];
    // The debug name is what disqualifies `var` from placeholder neutralization.
    module.debug_names = vec![Instruction::new(
        Op::Name,
        None,
        None,
        vec![
            Operand::IdRef(var),
            Operand::LiteralString("g_state".into()),
        ],
    )];

    let chain = 60;
    let chain_ops = vec![Operand::IdRef(var), Operand::IdRef(idx0)];
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(float),
            Some(40),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(41),
            ],
        )),
        parameters: vec![],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![Instruction::new(
                Op::AccessChain,
                Some(ptr_priv_float),
                Some(chain),
                chain_ops.clone(),
            )],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });

    let mut ctx = crate::passes::Ctx::new(module);
    run_idempotent(&mut ctx, |c| {
        neutralize_private_placeholder_access_chains(c, 0).unwrap()
    });

    let body = &ctx.module.functions[0].blocks[0].instructions;
    assert_eq!(body.len(), 1, "no instruction is added or removed");
    assert_eq!(body[0].class.opcode, Op::AccessChain);
    assert_eq!(body[0].operands, chain_ops, "the named chain is untouched");
}

// Store-to-load forwarding of an inlined local pointer field, keyed on the emitter's typed sidecar.
// A stored source through an access chain with key {root, indices} is matched to a later load carrying
// the same key, and every use of that load is repointed at the stored source.
#[test]
fn recover_inlined_local_pointer_fields_forwards_stored_source_to_matching_load() {
    let float = 1;
    let ptr_sb_float = 2;
    let uint = 3;
    let c1 = 4;
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_float),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(float),
            ],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(c1),
            vec![Operand::LiteralBit32(1)],
        ),
    ];

    let root = 50; // access-chain base -> key root
    let source = 80; // the real pointer the field should resolve to
    let stored_val = 70; // the value stored, marked with its source
    let chain = 60; // access chain, key {root:50, indices:[1]}
    let load = 90; // load marked with the matching key
    let consumer = 100; // uses the load result
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(float),
            Some(40),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(41),
            ],
        )),
        parameters: vec![
            Instruction::new(
                Op::FunctionParameter,
                Some(ptr_sb_float),
                Some(root),
                vec![],
            ),
            Instruction::new(
                Op::FunctionParameter,
                Some(ptr_sb_float),
                Some(source),
                vec![],
            ),
            Instruction::new(
                Op::FunctionParameter,
                Some(ptr_sb_float),
                Some(stored_val),
                vec![],
            ),
        ],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::AccessChain,
                    Some(ptr_sb_float),
                    Some(chain),
                    vec![Operand::IdRef(root), Operand::IdRef(c1)],
                ),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(chain), Operand::IdRef(stored_val)],
                ),
                Instruction::new(
                    Op::Load,
                    Some(ptr_sb_float),
                    Some(load),
                    vec![Operand::IdRef(chain)],
                ),
                Instruction::new(
                    Op::CopyObject,
                    Some(ptr_sb_float),
                    Some(consumer),
                    vec![Operand::IdRef(load)],
                ),
            ],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });
    let mut ctx = crate::passes::Ctx::new(module);
    ctx.emit_sidecar
        .local_pointer_field_stores
        .push(crate::emit_sidecar::LocalPointerFieldStore {
            id: stored_val,
            source,
        });
    ctx.emit_sidecar
        .local_pointer_field_loads
        .push(crate::emit_sidecar::LocalPointerFieldLoad {
            id: load,
            root,
            indices: vec![1],
        });
    run_idempotent(&mut ctx, |c| recover_inlined_local_pointer_fields(c, 0));

    let body = &ctx.module.functions[0].blocks[0].instructions;
    let consumer_inst = body.iter().find(|i| i.result_id == Some(consumer)).unwrap();
    assert_eq!(
        consumer_inst.operands,
        vec![Operand::IdRef(source)],
        "the load use is repointed at the stored source pointer"
    );
    // The load instruction itself keeps its id (only uses are rewritten).
    assert!(
        body.iter().any(|i| i.result_id == Some(load)),
        "the load def id is preserved; only its uses forward"
    );
}

// Forwarding only fires when the typed load fact's key matches a stored field. A load carrying a
// DIFFERENT key ({root, [2]} vs the stored {root, [1]}) is not forwarded, so its use is left intact.
#[test]
fn recover_inlined_local_pointer_fields_leaves_key_mismatched_load_alone() {
    let float = 1;
    let ptr_sb_float = 2;
    let uint = 3;
    let c1 = 4;
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(100));
    module.types_global_values = vec![
        Instruction::new(
            Op::TypeFloat,
            None,
            Some(float),
            vec![Operand::LiteralBit32(32)],
        ),
        Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_sb_float),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(float),
            ],
        ),
        Instruction::new(
            Op::TypeInt,
            None,
            Some(uint),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::Constant,
            Some(uint),
            Some(c1),
            vec![Operand::LiteralBit32(1)],
        ),
    ];

    let root = 50;
    let source = 80;
    let stored_val = 70;
    let chain = 60;
    let load = 90;
    let consumer = 100;
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(float),
            Some(40),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(41),
            ],
        )),
        parameters: vec![
            Instruction::new(
                Op::FunctionParameter,
                Some(ptr_sb_float),
                Some(root),
                vec![],
            ),
            Instruction::new(
                Op::FunctionParameter,
                Some(ptr_sb_float),
                Some(source),
                vec![],
            ),
            Instruction::new(
                Op::FunctionParameter,
                Some(ptr_sb_float),
                Some(stored_val),
                vec![],
            ),
        ],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(52), vec![])),
            instructions: vec![
                Instruction::new(
                    Op::AccessChain,
                    Some(ptr_sb_float),
                    Some(chain),
                    vec![Operand::IdRef(root), Operand::IdRef(c1)],
                ),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(chain), Operand::IdRef(stored_val)],
                ),
                Instruction::new(
                    Op::Load,
                    Some(ptr_sb_float),
                    Some(load),
                    vec![Operand::IdRef(chain)],
                ),
                Instruction::new(
                    Op::CopyObject,
                    Some(ptr_sb_float),
                    Some(consumer),
                    vec![Operand::IdRef(load)],
                ),
            ],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });
    let mut ctx = crate::passes::Ctx::new(module);
    ctx.emit_sidecar
        .local_pointer_field_stores
        .push(crate::emit_sidecar::LocalPointerFieldStore {
            id: stored_val,
            source,
        });
    ctx.emit_sidecar
        .local_pointer_field_loads
        .push(crate::emit_sidecar::LocalPointerFieldLoad {
            id: load,
            root,
            indices: vec![2],
        });
    run_idempotent(&mut ctx, |c| recover_inlined_local_pointer_fields(c, 0));

    let body = &ctx.module.functions[0].blocks[0].instructions;
    let consumer_inst = body.iter().find(|i| i.result_id == Some(consumer)).unwrap();
    assert_eq!(
        consumer_inst.operands,
        vec![Operand::IdRef(load)],
        "the key-mismatched load is not forwarded"
    );
}
