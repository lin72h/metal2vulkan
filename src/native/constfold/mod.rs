//! Static constant-branch pruning (FC-default dead-code elimination).
//!
//! Metal `[[function_constant]]`-gated optional features (an optional `mask` buffer, a
//! `do_causal` flag, ...) compile to AIR that loads a function-constant predicate global, branches
//! on it, and — in the not-taken arm — dereferences the optional buffer. We do NOT plumb runtime
//! function-constant specialization, so we model every function constant at its DISABLED DEFAULT
//! (the AIR `air.fc_initializer` global is `ConstantNull`). The optional buffer therefore has no
//! descriptor binding; `build_stage_input` demotes its unmodeled pointer param to a zero-init Private
//! placeholder SCALAR, and the dead arm's GEPs over-index that scalar -> spirv-val "reached
//! non-composite". The golden was captured with the SAME defaults, so the dead arm genuinely never
//! executes: statically pruning it (and the placeholder GEPs it feeds, even through pointer phis) is
//! semantics-preserving by construction.
//!
//! This is a small, GENERAL optimizer — constant propagation + conditional-branch folding +
//! unreachable-block removal + trivial-phi collapse + dead-code elimination, to a fixpoint. It keys
//! on NOTHING workload-specific: the only seed is "a scalar-int/bool global that is never stored and
//! has a constant initializer" plus "a scalar global stored exactly once (in the entry block, with no
//! preceding load) by a constant value" — both of which are the function-constant machinery the
//! emitter produced, identified structurally. Primary construction folds the constant-controlled
//! CFG without global DCE; raw-CFG construction retains the full DCE form for values that require
//! it. Neither representation depends on validator wording to decide which CFG is executable.

mod prune;
pub(in crate::native) use prune::*;
mod constants;
pub(in crate::native) use constants::*;
mod nonzero;
pub(in crate::native) use nonzero::*;
mod eval;
pub(in crate::native) use eval::*;
mod simplify;
pub(in crate::native) use simplify::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::Operand;
    use crate::spirv_module::{Block, Function, Instruction, Module, ModuleHeader};
    use spirv::{Op, StorageClass, Word};

    fn inst(op: Op, ty: Option<Word>, res: Option<Word>, ops: Vec<Operand>) -> Instruction {
        Instruction::new(op, ty, res, ops)
    }

    // A function-constant predicate computed as `(fc >> 0) & 1` over a disabled-default
    // (`OpConstantNull` = 0) FC global must fold to 0, so the arm it gates is statically dead and
    // pruned. This exercises the `OpBitwiseAnd`/`OpShiftRightLogical` modeling in `forward_eval`
    // (the `air.normalize_function_constant_predicate` lowering shape): without it the predicate
    // stays unknown and the dead arm survives.
    #[test]
    fn prunes_dead_arm_gated_by_bitwise_fc_predicate() {
        // ids: uint=1 bool=2 ptrPrivUint=3 | uint_0=10 uint_1=11 null=12 | fc_global=20
        //      labels: entry=30 then=34 else=35 merge=36 | %31 load %32 shr %33 and %38 ieq
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(40));
        m.types_global_values = vec![
            inst(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            inst(Op::TypeBool, None, Some(2), vec![]),
            inst(
                Op::TypePointer,
                None,
                Some(3),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(1),
                ],
            ),
            inst(
                Op::Constant,
                Some(1),
                Some(10),
                vec![Operand::LiteralBit32(0)],
            ),
            inst(
                Op::Constant,
                Some(1),
                Some(11),
                vec![Operand::LiteralBit32(1)],
            ),
            inst(Op::ConstantNull, Some(1), Some(12), vec![]),
            inst(
                Op::Variable,
                Some(3),
                Some(20),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(12),
                ],
            ),
        ];
        let entry = Block {
            label: Some(inst(Op::Label, None, Some(30), vec![])),
            instructions: vec![
                inst(Op::Load, Some(1), Some(31), vec![Operand::IdRef(20)]),
                // %32 = (fc >> 0)
                inst(
                    Op::ShiftRightLogical,
                    Some(1),
                    Some(32),
                    vec![Operand::IdRef(31), Operand::IdRef(10)],
                ),
                // %33 = %32 & 1
                inst(
                    Op::BitwiseAnd,
                    Some(1),
                    Some(33),
                    vec![Operand::IdRef(32), Operand::IdRef(11)],
                ),
                // %38 = (%33 == 0)  -> true, so the conditional takes %then and %else is dead
                inst(
                    Op::IEqual,
                    Some(2),
                    Some(38),
                    vec![Operand::IdRef(33), Operand::IdRef(10)],
                ),
                inst(
                    Op::SelectionMerge,
                    None,
                    None,
                    vec![
                        Operand::IdRef(36),
                        Operand::SelectionControl(spirv::SelectionControl::NONE),
                    ],
                ),
                inst(
                    Op::BranchConditional,
                    None,
                    None,
                    vec![Operand::IdRef(38), Operand::IdRef(34), Operand::IdRef(35)],
                ),
            ],
        };
        let then_b = Block {
            label: Some(inst(Op::Label, None, Some(34), vec![])),
            instructions: vec![inst(Op::Branch, None, None, vec![Operand::IdRef(36)])],
        };
        let else_b = Block {
            label: Some(inst(Op::Label, None, Some(35), vec![])),
            instructions: vec![inst(Op::Branch, None, None, vec![Operand::IdRef(36)])],
        };
        let merge_b = Block {
            label: Some(inst(Op::Label, None, Some(36), vec![])),
            instructions: vec![inst(Op::Return, None, None, vec![])],
        };
        let mut func = Function::new();
        func.blocks = vec![entry, then_b, else_b, merge_b];
        m.functions = vec![func];
        m.debug_names = vec![
            inst(
                Op::Name,
                None,
                None,
                vec![
                    Operand::IdRef(35),
                    Operand::LiteralString("dead-arm".into()),
                ],
            ),
            inst(
                Op::Name,
                None,
                None,
                vec![Operand::IdRef(39), Operand::LiteralString("unowned".into())],
            ),
        ];

        crate::native::rewrites::prune_constant_branches_module(&mut m).expect("expected a fold");
        let labels: Vec<Word> = m.functions[0]
            .blocks
            .iter()
            .filter_map(|b| b.label.as_ref().and_then(|l| l.result_id))
            .collect();
        assert!(
            !labels.contains(&35),
            "dead else-arm (%35) should be pruned: {labels:?}"
        );
        assert!(
            labels.contains(&34),
            "taken then-arm (%34) should survive: {labels:?}"
        );
        assert_eq!(m.debug_names.len(), 1);
        assert_eq!(m.debug_names[0].operands[0], Operand::IdRef(39));
    }

    #[test]
    fn keeps_loop_exit_gated_by_loop_carried_phi() {
        // A loop induction phi has one constant entry arm and one backedge defined by the loop body.
        // The backedge is not known during the first SCCP sweep; treating that unknown as neutral
        // folds `%i` to 0, then `%next < 48` to true, and incorrectly deletes the exit.
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(70));
        m.types_global_values = vec![
            inst(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            inst(Op::TypeBool, None, Some(2), vec![]),
            inst(
                Op::Constant,
                Some(1),
                Some(10),
                vec![Operand::LiteralBit32(0)],
            ),
            inst(
                Op::Constant,
                Some(1),
                Some(11),
                vec![Operand::LiteralBit32(1)],
            ),
            inst(
                Op::Constant,
                Some(1),
                Some(12),
                vec![Operand::LiteralBit32(48)],
            ),
        ];
        let entry = Block {
            label: Some(inst(Op::Label, None, Some(30), vec![])),
            instructions: vec![inst(Op::Branch, None, None, vec![Operand::IdRef(31)])],
        };
        let header = Block {
            label: Some(inst(Op::Label, None, Some(31), vec![])),
            instructions: vec![
                inst(
                    Op::Phi,
                    Some(1),
                    Some(50),
                    vec![
                        Operand::IdRef(10),
                        Operand::IdRef(30),
                        Operand::IdRef(51),
                        Operand::IdRef(32),
                    ],
                ),
                inst(
                    Op::IAdd,
                    Some(1),
                    Some(51),
                    vec![Operand::IdRef(50), Operand::IdRef(11)],
                ),
                inst(
                    Op::ULessThan,
                    Some(2),
                    Some(52),
                    vec![Operand::IdRef(51), Operand::IdRef(12)],
                ),
                inst(
                    Op::LoopMerge,
                    None,
                    None,
                    vec![
                        Operand::IdRef(33),
                        Operand::IdRef(32),
                        Operand::LoopControl(spirv::LoopControl::NONE),
                    ],
                ),
                inst(
                    Op::BranchConditional,
                    None,
                    None,
                    vec![Operand::IdRef(52), Operand::IdRef(32), Operand::IdRef(33)],
                ),
            ],
        };
        let latch = Block {
            label: Some(inst(Op::Label, None, Some(32), vec![])),
            instructions: vec![inst(Op::Branch, None, None, vec![Operand::IdRef(31)])],
        };
        let exit = Block {
            label: Some(inst(Op::Label, None, Some(33), vec![])),
            instructions: vec![inst(Op::Return, None, None, vec![])],
        };
        let mut func = Function::new();
        func.blocks = vec![entry, header, latch, exit];
        m.functions = vec![func];

        prune_constant_branches(&mut m);

        let labels: Vec<Word> = m.functions[0]
            .blocks
            .iter()
            .filter_map(|b| b.label.as_ref().and_then(|l| l.result_id))
            .collect();
        assert!(
            labels.contains(&33),
            "loop exit must not be pruned from an unknown induction phi: {labels:?}"
        );
        let header_term = m.functions[0].blocks[1]
            .instructions
            .last()
            .expect("header terminator");
        assert_eq!(header_term.class.opcode, Op::BranchConditional);
    }

    // A self-referential DEAD pointer-induction cycle — `%50 = OpPhi [%51 latch] [%base pre]` whose
    // only remaining reference is its own back-edge `%51 = OpPtrAccessChain %50` (the consumer load
    // pruned with a dead arm) — must be collected by DCE. The naive "any operand reference = used"
    // mark keeps it alive forever (phi marks the access chain used, the access chain marks the phi
    // used); transitive liveness from sinks drops the whole cycle. With an external use it stays.
    fn module_with_ptr_cycle(extra_use: bool) -> Module {
        // ids: float=1 ptrUC_float=2 | base=12 uint_1=11 | pre=30 hdr=31 latch=32 exit=33
        //      phi=50 step=51 | sink load=60
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(70));
        m.types_global_values = vec![
            inst(
                Op::TypeFloat,
                None,
                Some(1),
                vec![Operand::LiteralBit32(32)],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(2),
                vec![
                    Operand::StorageClass(StorageClass::UniformConstant),
                    Operand::IdRef(1),
                ],
            ),
            inst(
                Op::Constant,
                Some(1),
                Some(11),
                vec![Operand::LiteralBit32(1)],
            ),
            inst(Op::ConstantNull, Some(2), Some(12), vec![]),
            // Private float* type (%3) + a Private sink variable (%20) for the live-case store.
            inst(
                Op::TypePointer,
                None,
                Some(3),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(1),
                ],
            ),
            inst(
                Op::Variable,
                Some(3),
                Some(20),
                vec![Operand::StorageClass(StorageClass::Private)],
            ),
        ];
        let pre = Block {
            label: Some(inst(Op::Label, None, Some(30), vec![])),
            instructions: vec![inst(Op::Branch, None, None, vec![Operand::IdRef(31)])],
        };
        let hdr = Block {
            label: Some(inst(Op::Label, None, Some(31), vec![])),
            instructions: vec![
                inst(
                    Op::Phi,
                    Some(2),
                    Some(50),
                    vec![
                        Operand::IdRef(51),
                        Operand::IdRef(32),
                        Operand::IdRef(12),
                        Operand::IdRef(30),
                    ],
                ),
                inst(Op::Branch, None, None, vec![Operand::IdRef(32)]),
            ],
        };
        let mut latch_insts = vec![inst(
            Op::PtrAccessChain,
            Some(2),
            Some(51),
            vec![Operand::IdRef(50), Operand::IdRef(11)],
        )];
        if extra_use {
            // A genuine sink: load through the cycle pointer and STORE the value to a Private global
            // (a non-pure use), so the whole cycle is reachable from a sink and must survive.
            latch_insts.push(inst(Op::Load, Some(1), Some(60), vec![Operand::IdRef(50)]));
            latch_insts.push(inst(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(20), Operand::IdRef(60)],
            ));
        }
        latch_insts.push(inst(Op::Branch, None, None, vec![Operand::IdRef(31)]));
        let latch = Block {
            label: Some(inst(Op::Label, None, Some(32), vec![])),
            instructions: latch_insts,
        };
        let mut func = Function::new();
        func.blocks = vec![pre, hdr, latch];
        m.functions = vec![func];
        m
    }

    #[test]
    fn dce_collects_dead_pointer_induction_cycle() {
        let mut m = module_with_ptr_cycle(false);
        assert!(dce(&mut m), "expected the dead cycle to be removed");
        let results: Vec<Word> = m.functions[0]
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter().filter_map(|i| i.result_id))
            .collect();
        assert!(
            !results.contains(&50) && !results.contains(&51),
            "dead self-referential pointer cycle (%50/%51) should be gone: {results:?}"
        );
    }

    #[test]
    fn dce_keeps_live_pointer_induction_cycle() {
        let mut m = module_with_ptr_cycle(true);
        dce(&mut m);
        let results: Vec<Word> = m.functions[0]
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter().filter_map(|i| i.result_id))
            .collect();
        assert!(
            results.contains(&50) && results.contains(&51),
            "a cycle with an external consumer (%60) must survive: {results:?}"
        );
    }

    // A guard `C[0] == 0` over a VECTOR function constant whose disabled default is a null vector
    // (`OpConstantNull %v4ushort`) must fold: element 0 of the null vector is 0, so the equality is
    // true and the arm it gates is dead. Exercises the composite-constant + `OpCompositeExtract`
    // modeling — without it the vector load stays opaque and the dead arm survives.
    #[test]
    fn prunes_dead_arm_gated_by_vector_fc_element() {
        // ids: ushort=1 bool=2 v4ushort=3 ptrPrivV4=4 | ushort_0=10 nullv4=11 | Cglobal=20
        //      labels entry=30 then=34 else=35 merge=36 | %31 load %32 extract %33 ieq
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(40));
        m.types_global_values = vec![
            inst(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(16), Operand::LiteralBit32(0)],
            ),
            inst(Op::TypeBool, None, Some(2), vec![]),
            inst(
                Op::TypeVector,
                None,
                Some(3),
                vec![Operand::IdRef(1), Operand::LiteralBit32(4)],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(4),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(3),
                ],
            ),
            inst(Op::ConstantNull, Some(1), Some(10), vec![]),
            inst(Op::ConstantNull, Some(3), Some(11), vec![]),
            inst(
                Op::Variable,
                Some(4),
                Some(20),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(11),
                ],
            ),
        ];
        let entry = Block {
            label: Some(inst(Op::Label, None, Some(30), vec![])),
            instructions: vec![
                inst(Op::Load, Some(3), Some(31), vec![Operand::IdRef(20)]),
                inst(
                    Op::CompositeExtract,
                    Some(1),
                    Some(32),
                    vec![Operand::IdRef(31), Operand::LiteralBit32(0)],
                ),
                inst(
                    Op::IEqual,
                    Some(2),
                    Some(33),
                    vec![Operand::IdRef(32), Operand::IdRef(10)],
                ),
                inst(
                    Op::SelectionMerge,
                    None,
                    None,
                    vec![
                        Operand::IdRef(36),
                        Operand::SelectionControl(spirv::SelectionControl::NONE),
                    ],
                ),
                inst(
                    Op::BranchConditional,
                    None,
                    None,
                    vec![Operand::IdRef(33), Operand::IdRef(34), Operand::IdRef(35)],
                ),
            ],
        };
        let then_b = Block {
            label: Some(inst(Op::Label, None, Some(34), vec![])),
            instructions: vec![inst(Op::Branch, None, None, vec![Operand::IdRef(36)])],
        };
        let else_b = Block {
            label: Some(inst(Op::Label, None, Some(35), vec![])),
            instructions: vec![inst(Op::Branch, None, None, vec![Operand::IdRef(36)])],
        };
        let merge_b = Block {
            label: Some(inst(Op::Label, None, Some(36), vec![])),
            instructions: vec![inst(Op::Return, None, None, vec![])],
        };
        let mut func = Function::new();
        func.blocks = vec![entry, then_b, else_b, merge_b];
        m.functions = vec![func];

        assert!(prune_constant_branches(&mut m), "expected a fold");
        let labels: Vec<Word> = m.functions[0]
            .blocks
            .iter()
            .filter_map(|b| b.label.as_ref().and_then(|l| l.result_id))
            .collect();
        assert!(
            !labels.contains(&35),
            "dead else-arm should be pruned: {labels:?}"
        );
        assert!(
            labels.contains(&34),
            "taken then-arm should survive: {labels:?}"
        );
    }

    // A grid-stride early-return guard `stride > stride - 1` folds to a taken return: `stride` is a
    // `NumWorkgroups`-derived value (>= 1 for any executing invocation), so `X > X-1` is statically
    // true and the guarded compute arm is dead. Exercises the nonzero analysis + affine self-minus-one
    // guard fold; without it the guard stays opaque and the arm survives.
    #[test]
    fn folds_grid_stride_early_return_guard() {
        // ids: uint=1 bool=2 v3uint=3 ptrInV3=4 | allones=10 (also a nonzero const seed) | nwg=20
        //      entry=30 body=34 ret=35 | %31 load %32 extract %33 iadd(X,-1) %38 ugt
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(40));
        m.types_global_values = vec![
            inst(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            inst(Op::TypeBool, None, Some(2), vec![]),
            inst(
                Op::TypeVector,
                None,
                Some(3),
                vec![Operand::IdRef(1), Operand::LiteralBit32(3)],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(4),
                vec![
                    Operand::StorageClass(StorageClass::Input),
                    Operand::IdRef(3),
                ],
            ),
            inst(
                Op::Constant,
                Some(1),
                Some(10),
                vec![Operand::LiteralBit32(0xFFFF_FFFF)],
            ),
            inst(
                Op::Variable,
                Some(4),
                Some(20),
                vec![Operand::StorageClass(StorageClass::Input)],
            ),
        ];
        m.annotations = vec![inst(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(20),
                Operand::Decoration(spirv::Decoration::BuiltIn),
                Operand::BuiltIn(spirv::BuiltIn::NumWorkgroups),
            ],
        )];
        let entry = Block {
            label: Some(inst(Op::Label, None, Some(30), vec![])),
            instructions: vec![
                inst(Op::Load, Some(3), Some(31), vec![Operand::IdRef(20)]),
                inst(
                    Op::CompositeExtract,
                    Some(1),
                    Some(32),
                    vec![Operand::IdRef(31), Operand::LiteralBit32(0)],
                ),
                // %33 = X + 0xFFFFFFFF = X - 1
                inst(
                    Op::IAdd,
                    Some(1),
                    Some(33),
                    vec![Operand::IdRef(32), Operand::IdRef(10)],
                ),
                // %38 = X > X-1 -> folds true, so %35 (return) is taken and %34 is dead
                inst(
                    Op::UGreaterThan,
                    Some(2),
                    Some(38),
                    vec![Operand::IdRef(32), Operand::IdRef(33)],
                ),
                inst(
                    Op::SelectionMerge,
                    None,
                    None,
                    vec![
                        Operand::IdRef(34),
                        Operand::SelectionControl(spirv::SelectionControl::NONE),
                    ],
                ),
                inst(
                    Op::BranchConditional,
                    None,
                    None,
                    vec![Operand::IdRef(38), Operand::IdRef(35), Operand::IdRef(34)],
                ),
            ],
        };
        let body_b = Block {
            label: Some(inst(Op::Label, None, Some(34), vec![])),
            instructions: vec![inst(Op::Branch, None, None, vec![Operand::IdRef(35)])],
        };
        let ret_b = Block {
            label: Some(inst(Op::Label, None, Some(35), vec![])),
            instructions: vec![inst(Op::Return, None, None, vec![])],
        };
        let mut func = Function::new();
        func.blocks = vec![entry, body_b, ret_b];
        m.functions = vec![func];

        assert!(
            prune_constant_branches(&mut m),
            "expected the guard to fold"
        );
        let labels: Vec<Word> = m.functions[0]
            .blocks
            .iter()
            .filter_map(|b| b.label.as_ref().and_then(|l| l.result_id))
            .collect();
        assert!(
            !labels.contains(&34),
            "dead compute arm (%34) should be pruned: {labels:?}"
        );
    }

    // A grid-bounds guard `tile < (W*H)` where the scalar dimension FCs W,H fold to 0 (disabled
    // default) must fold: `x < 0` is false for every unsigned x, so the compute arm it gates is dead
    // — even though `tile` is a runtime (unknown) dispatch value. Exercises the one-sided unsigned
    // bound in `ucmp_fold`; without it the guard stays opaque and the FC-zero-work MXU arm survives.
    #[test]
    fn folds_unsigned_less_than_zero_bounds_guard() {
        // ids: uint=1 bool=2 ptrPrivU=3 ptrInU=4 | null=10 | Wg=20 Hg=21 tileVar=22
        //      entry=30 compute=34 skip=35 | %31 loadW %32 loadH %33 mul %36 loadTile %38 ult
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(40));
        m.types_global_values = vec![
            inst(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            inst(Op::TypeBool, None, Some(2), vec![]),
            inst(
                Op::TypePointer,
                None,
                Some(3),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(1),
                ],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(4),
                vec![
                    Operand::StorageClass(StorageClass::Input),
                    Operand::IdRef(1),
                ],
            ),
            inst(Op::ConstantNull, Some(1), Some(10), vec![]),
            inst(
                Op::Variable,
                Some(3),
                Some(20),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(10),
                ],
            ),
            inst(
                Op::Variable,
                Some(3),
                Some(21),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(10),
                ],
            ),
            inst(
                Op::Variable,
                Some(4),
                Some(22),
                vec![Operand::StorageClass(StorageClass::Input)],
            ),
        ];
        let entry = Block {
            label: Some(inst(Op::Label, None, Some(30), vec![])),
            instructions: vec![
                inst(Op::Load, Some(1), Some(31), vec![Operand::IdRef(20)]),
                inst(Op::Load, Some(1), Some(32), vec![Operand::IdRef(21)]),
                inst(
                    Op::IMul,
                    Some(1),
                    Some(33),
                    vec![Operand::IdRef(31), Operand::IdRef(32)],
                ),
                inst(Op::Load, Some(1), Some(36), vec![Operand::IdRef(22)]),
                // %38 = tile < (W*H) = tile < 0 -> false, so %35 (skip) is taken and %34 is dead
                inst(
                    Op::ULessThan,
                    Some(2),
                    Some(38),
                    vec![Operand::IdRef(36), Operand::IdRef(33)],
                ),
                inst(
                    Op::SelectionMerge,
                    None,
                    None,
                    vec![
                        Operand::IdRef(35),
                        Operand::SelectionControl(spirv::SelectionControl::NONE),
                    ],
                ),
                inst(
                    Op::BranchConditional,
                    None,
                    None,
                    vec![Operand::IdRef(38), Operand::IdRef(34), Operand::IdRef(35)],
                ),
            ],
        };
        let compute_b = Block {
            label: Some(inst(Op::Label, None, Some(34), vec![])),
            instructions: vec![inst(Op::Branch, None, None, vec![Operand::IdRef(35)])],
        };
        let skip_b = Block {
            label: Some(inst(Op::Label, None, Some(35), vec![])),
            instructions: vec![inst(Op::Return, None, None, vec![])],
        };
        let mut func = Function::new();
        func.blocks = vec![entry, compute_b, skip_b];
        m.functions = vec![func];

        assert!(
            prune_constant_branches(&mut m),
            "expected the guard to fold"
        );
        let labels: Vec<Word> = m.functions[0]
            .blocks
            .iter()
            .filter_map(|b| b.label.as_ref().and_then(|l| l.result_id))
            .collect();
        assert!(
            !labels.contains(&34),
            "dead compute arm (%34) should be pruned: {labels:?}"
        );
    }
}
