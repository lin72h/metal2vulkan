//! Differential tests: construct a function's control flow and require it to still compute what it
//! computed before.

use super::{assert_construction_preserves_semantics, interp, nests, outcomes, shapes, STEP_LIMIT};
use crate::native::cfg_testkit::build::CfgBuilder;
use crate::native::cfg_testkit::interp::{Outcome, Value};

/// A spread of arguments per shape. The generated conditions are masks over the accumulator, so
/// different low bits take genuinely different paths.
const ARGUMENTS: &[&[u32]] = &[&[0], &[1], &[2], &[3], &[7], &[16], &[31], &[1000]];

/// Sum `0..n` with a counted loop, and require the interpreter to agree with the closed form.
#[test]
fn the_interpreter_agrees_with_the_closed_form_of_a_counted_sum() {
    let module = counted_sum();
    for n in [0u32, 1, 2, 5, 17, 64] {
        let expected = (0..n).sum::<u32>();
        assert_eq!(
            interp::run_module(&module, &[n], STEP_LIMIT).expect("counted sum"),
            Outcome::Returned(Value::Int(expected)),
            "sum of 0..{n}"
        );
    }
}

/// A function that never leaves its loop must be reported, not run forever. This is the guard that
/// turns "the construction lost the exit edge" from a hung test process into a failure.
#[test]
fn the_interpreter_reports_a_function_that_does_not_terminate() {
    let mut builder = CfgBuilder::new(1);
    let one = builder.constant(1);
    builder.block("entry");
    builder.branch("header");
    builder.block("header");
    let counter = builder.reserve_value();
    let seed = builder.parameter(0);
    let merged = builder.phi(&[(seed, "entry"), (counter, "header")]);
    builder.add_into(counter, merged, one);
    builder.branch("header");
    let module = builder.finish();

    let error = interp::run_module(&module, &[0], 10_000).expect_err("no exit edge");
    assert!(error.contains("does not terminate"), "{error}");
}

/// `n * (n - 1) / 2` by accumulation, as an independently checkable oracle subject.
fn counted_sum() -> crate::spirv_module::Module {
    let mut builder = CfgBuilder::new(1);
    let zero = builder.constant(0);
    let one = builder.constant(1);

    builder.block("entry");
    builder.branch("header");

    builder.block("header");
    let total = builder.reserve_value();
    let index = builder.reserve_value();
    let limit = builder.parameter(0);
    let total_in = builder.phi(&[(zero, "entry"), (total, "body")]);
    let index_in = builder.phi(&[(zero, "entry"), (index, "body")]);
    let below = builder.less_than(index_in, limit);
    builder.branch_conditional(below, "body", "exit");

    builder.block("body");
    builder.add_into(total, total_in, index_in);
    builder.add_into(index, index_in, one);
    builder.branch("header");

    builder.block("exit");
    builder.return_value(total_in);

    builder.finish()
}

/// Every generated shape must terminate on its own, before it is ever handed to a structurizer.
/// If this fails the generator is broken, not the product.
#[test]
fn generated_shapes_terminate_before_construction() {
    for seed in 0..64u64 {
        let shape = shapes::shape(seed, 4);
        let module = shapes::author(&shape);
        let results = outcomes(&module, ARGUMENTS);
        assert_eq!(results.len(), ARGUMENTS.len(), "seed {seed}");
    }
}

/// The generator has to actually produce the shape it claims to: unstructured control flow, which
/// is what selects the constructors in the first place.
#[test]
fn generated_shapes_are_unstructured_enough_to_select_construction() {
    let mut selected = 0;
    for seed in 0..64u64 {
        let module = shapes::author(&shapes::shape(seed, 4));
        let function = &module.functions[0];
        if crate::native::rewrites::blocks_have_unowned_selection_header(&function.blocks)
            || crate::native::rewrites::function_has_unowned_backedge(function)
        {
            selected += 1;
        }
    }
    assert!(
        selected >= 60,
        "only {selected}/64 generated shapes need construction; the generator has stopped \
         producing unstructured control flow and the differential sweep is testing nothing"
    );
}

/// The differential sweep. Construct each generated shape and require it to compute the same
/// answer on every argument.
#[test]
fn construction_preserves_semantics_across_generated_shapes() {
    for seed in 0..64u64 {
        let shape = shapes::shape(seed, 4);
        let module = shapes::author(&shape);
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_construction_preserves_semantics(module, ARGUMENTS);
        }))
        .unwrap_or_else(|_| {
            panic!(
                "construction lost the semantics of this shape:\n{}",
                shapes::describe(&shape)
            )
        });
    }
}

/// The same sweep at a size where the nesting structurizer has more to get wrong.
#[test]
fn construction_preserves_semantics_across_deeper_generated_shapes() {
    for seed in 1000..1032u64 {
        let shape = shapes::shape(seed, 6);
        let module = shapes::author(&shape);
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_construction_preserves_semantics(module, ARGUMENTS);
        }))
        .unwrap_or_else(|_| {
            panic!(
                "construction lost the semantics of this shape:\n{}",
                shapes::describe(&shape)
            )
        });
    }
}

/// Regression: a staged edge that leaves a selection *and* a loop on the way to the function's
/// exit must name its destination in the flow variable.
///
/// `while (c) { if (d) { if (e) break; } q(); }` — the `break` leaves the inner selection, the
/// outer selection, and the loop. The nesting emitter routes it out through the innermost
/// construct's merge, which is a dispatch that reads the flow variable to decide whether to enter
/// `q()` or forward on out of the loop. The edge used to decide whether to write that variable by
/// asking the construct that owns the *destination* — the loop — rather than the merge it was
/// about to branch to, so on this one edge it wrote nothing and the dispatch read whatever the
/// previous iteration had left there. The result was valid SPIR-V that ran `q()` after a `break`.
#[test]
fn a_break_out_of_two_constructs_names_its_destination() {
    let module = break_out_of_a_selection_inside_a_loop();
    // Nesting has to be what answers this shape. Declining it is safe -- the flow-variable check
    // in `reloop_nest::verify` turns the defect into a decline -- but a decline hands the function
    // to the state machine, which is the flattened shape a driver's compiler chokes on. Pin both:
    // the shape nests, and the nesting computes what it used to.
    assert!(
        nests(&module),
        "the nesting structurizer declined a break out of two constructs; the function falls back \
         to the whole-function dispatch this pass exists to avoid"
    );
    // Both bits of the argument matter: bit 2 selects `d`, bit 0 selects `e`, and only the
    // combination that takes the `break` on a later iteration exposes a stale flow value.
    assert_construction_preserves_semantics(module, ARGUMENTS);
}

/// Semantics alone cannot tell a correct nesting from a decline, because a decline is also
/// correct. This pins how much of the generated family the nesting actually takes, so a change
/// that buys correctness by declining everything fails here rather than passing quietly.
#[test]
fn most_generated_shapes_nest_instead_of_falling_back_to_the_state_machine() {
    let mut nested = 0;
    let mut total = 0;
    for (seeds, depth) in [(0..64u64, 4u32), (1000..1064, 6)] {
        for seed in seeds {
            total += 1;
            if nests(&shapes::author(&shapes::shape(seed, depth))) {
                nested += 1;
            }
        }
    }
    // Measured at 100/128 when this was written; the rest are shapes the emitter genuinely cannot
    // express -- a `continue` of an outer loop is the common one. The floor is set below that, not
    // at it, so ordinary shape-tree churn does not fail the test, but losing a family does.
    assert!(
        nested * 10 >= total * 7,
        "only {nested}/{total} generated reducible shapes nest; the rest fall back to the \
         whole-function dispatch that hangs a driver's shader compiler"
    );
}

/// `while (acc < bound) { if (d) { if (e) break; } q(); }`, with the accumulator threaded through
/// so a wrong route changes the answer.
fn break_out_of_a_selection_inside_a_loop() -> crate::spirv_module::Module {
    let mut builder = CfgBuilder::new(1);
    let (zero, one, four) = (
        builder.constant(0),
        builder.constant(1),
        builder.constant(4),
    );
    let bound = builder.constant(4034);
    let (three, five, six, seven, two) = (
        builder.constant(3),
        builder.constant(5),
        builder.constant(6),
        builder.constant(7),
        builder.constant(2),
    );

    let entering = builder.reserve_value();
    let latched = builder.reserve_value();

    builder.block("entry");
    let seed = builder.parameter(0);
    let start = builder.add(seed, one);
    builder.branch("header");

    builder.block("header");
    let merged = builder.phi(&[(start, "entry"), (latched, "latch")]);
    builder.add_into(entering, merged, three);
    let below = builder.less_than(entering, bound);
    builder.branch_conditional(below, "body", "exit");

    builder.block("body");
    let in_body = builder.add(entering, four);
    let masked = builder.bitwise_and(in_body, four);
    let d = builder.equal(masked, zero);
    builder.branch_conditional(d, "guard", "skip");

    builder.block("guard");
    let in_guard = builder.add(in_body, six);
    let low = builder.bitwise_and(in_guard, one);
    let e = builder.equal(low, zero);
    // The `break`: out of this selection, out of the one at `body`, and out of the loop.
    builder.branch_conditional(e, "exit", "latch");

    builder.block("skip");
    let in_skip = builder.add(in_body, seven);
    builder.branch("latch");

    builder.block("latch");
    let carried = builder.phi(&[(in_guard, "guard"), (in_skip, "skip")]);
    builder.add_into(latched, carried, five);
    builder.branch("header");

    builder.block("exit");
    let leaving = builder.phi(&[(entering, "header"), (in_guard, "guard")]);
    let result = builder.add(leaving, two);
    builder.return_value(result);

    builder.finish()
}

/// The state-machine constructor's own territory. A graph with an edge into the interior of an
/// already-written region is irreducible, so the nesting structurizer declines it by contract and
/// the state machine is what has to be right.
#[test]
fn construction_preserves_semantics_across_irreducible_shapes() {
    for depth in [3u32, 5] {
        for seed in 0..48u64 {
            let shape = shapes::irreducible_shape(seed, depth, 3);
            let module = shapes::author(&shape);
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                assert_construction_preserves_semantics(module, ARGUMENTS);
            }))
            .unwrap_or_else(|_| {
                panic!(
                    "construction lost the semantics of this shape:\n{}",
                    shapes::describe(&shape)
                )
            });
        }
    }
}

/// The irreducible family only tests the state machine if it really is irreducible. Nesting
/// declining most of it is the observable form of that; if this drops, the sweep above has quietly
/// become a second run of the reducible one.
#[test]
fn irreducible_shapes_are_declined_by_the_nesting_structurizer() {
    let mut declined = 0;
    let mut total = 0;
    for depth in [3u32, 5] {
        for seed in 0..48u64 {
            total += 1;
            if !nests(&shapes::author(&shapes::irreducible_shape(seed, depth, 3))) {
                declined += 1;
            }
        }
    }
    // Measured at 66/96 when this was written. A crossing whose target happens to dominate its
    // source is an ordinary back edge and leaves the graph reducible, so the family is a mixture
    // by construction; the floor only has to be high enough to prove the mixture is real.
    assert!(
        declined * 2 >= total,
        "only {declined}/{total} shapes with interior cross edges reach the state machine"
    );
}

/// Two phis at a loop header that read each other: `x = phi(x0, y)` and `y = phi(y0, x)`. Both
/// incoming values are the header's own results, so the pair swaps every iteration. Demoting them
/// to memory turns a simultaneous assignment into two sequential stores, and doing that in the
/// obvious order writes the new x before y has read the old one.
#[test]
fn a_pair_of_loop_phis_that_swap_still_swaps() {
    let mut builder = CfgBuilder::new(1);
    let (one, two, three, bound) = (
        builder.constant(1),
        builder.constant(2),
        builder.constant(3),
        builder.constant(64),
    );

    let x_next = builder.reserve_value();
    let y_next = builder.reserve_value();
    let i_next = builder.reserve_value();

    builder.block("entry");
    let seed = builder.parameter(0);
    let x0 = builder.add(seed, one);
    let y0 = builder.add(seed, two);
    builder.branch("header");

    builder.block("header");
    // The swap: x takes what y had, y takes what x had.
    let x = builder.phi(&[(x0, "entry"), (y_next, "latch")]);
    let y = builder.phi(&[(y0, "entry"), (x_next, "latch")]);
    let i = builder.phi(&[(seed, "entry"), (i_next, "latch")]);
    let below = builder.less_than(i, bound);
    builder.branch_conditional(below, "latch", "exit");

    builder.block("latch");
    // Only x advances, so x and y are distinguishable after any number of swaps.
    builder.add_into(x_next, x, three);
    builder.add_into(y_next, y, three);
    builder.add_into(i_next, i, one);
    builder.branch("header");

    builder.block("exit");
    let mixed = builder.bitwise_and(x, one);
    let combined = builder.add(y, mixed);
    builder.return_value(combined);

    assert_construction_preserves_semantics(builder.finish(), ARGUMENTS);
}

/// The generated family only exercises the phi-demotion hazard if its shapes actually have back
/// edges for [`shapes::author`] to hang the value swap on. If this drops, the sweeps above have
/// quietly become straight-line value flow.
#[test]
fn generated_shapes_carry_the_phi_swap_on_real_back_edges() {
    let mut swapping = 0;
    let mut shapes_with_a_swap = 0;
    let mut total = 0;
    for (seeds, depth, crossings) in [(0..64u64, 4u32, 0u32), (0..48, 5, 3)] {
        for seed in seeds {
            total += 1;
            let edges = shapes::irreducible_shape(seed, depth, crossings).swapping_edges();
            swapping += edges;
            if edges > 0 {
                shapes_with_a_swap += 1;
            }
        }
    }
    // Measured at 105/112 shapes carrying 262 swapping edges when this was written.
    assert!(
        shapes_with_a_swap * 2 >= total && swapping >= total,
        "only {shapes_with_a_swap}/{total} generated shapes have a back edge ({swapping} edges \
         in total); the swap the author hangs on them is not being exercised"
    );
}

/// Constant folding is a rewrite like any other, and the only thing that can say whether it
/// preserved meaning is running the module. A function seeded entirely from a constant global is
/// statically decidable, so `constfold` propagates, folds branches, deletes the blocks that become
/// unreachable, and collapses the phis that lose predecessors — four rewrites whose interaction is
/// where a fold goes wrong, over control flow far messier than an authored fixture would be.
///
/// Both pass orders are checked. Folding a constructed CFG and constructing a folded one are both
/// things the pipeline does, and an optimizer that is right on the input shape can still be wrong
/// on the shape a structurizer left behind.
#[test]
fn constant_folding_preserves_the_one_answer_a_constant_seeded_shape_has() {
    for depth in [3u32, 5] {
        for seed in 0..32u64 {
            for value in [0u32, 3] {
                let shape = shapes::irreducible_shape(seed, depth, 2);
                let (module, slot) = shapes::author_constant_seeded(&shape, value);
                let expected = interp::run_module_to_global(&module, slot, STEP_LIMIT)
                    .unwrap_or_else(|error| {
                        panic!("authored: {error}\n{}", shapes::describe(&shape))
                    });

                let mut folded_then_constructed = module.clone();
                crate::native::rewrites::prune_constant_cfg_module_if_changed(
                    &mut folded_then_constructed,
                );
                let _ = crate::native::rewrites::construct_cfg_functions_module(
                    &mut folded_then_constructed,
                    &std::collections::HashSet::new(),
                );

                let mut constructed_then_folded = module;
                let _ = crate::native::rewrites::construct_cfg_functions_module(
                    &mut constructed_then_folded,
                    &std::collections::HashSet::new(),
                );
                crate::native::rewrites::prune_constant_cfg_module_if_changed(
                    &mut constructed_then_folded,
                );

                for (order, module) in [
                    ("fold then construct", &folded_then_constructed),
                    ("construct then fold", &constructed_then_folded),
                ] {
                    let actual = interp::run_module_to_global(module, slot, STEP_LIMIT)
                        .unwrap_or_else(|error| {
                            panic!(
                                "{order} (value {value}): {error}\n{}",
                                shapes::describe(&shape)
                            )
                        });
                    assert_eq!(
                        actual,
                        expected,
                        "{order} (value {value}) changed the answer of this shape:\n{}",
                        shapes::describe(&shape)
                    );
                }
            }
        }
    }
}

/// The sweep above only means something if the fold is actually folding. A constant-seeded shape's
/// straight-line prefix is decidable, so blocks must disappear; a loop-carried condition is not,
/// so not all of them do.
#[test]
fn constant_folding_actually_removes_blocks_from_a_constant_seeded_shape() {
    let mut before = 0;
    let mut after = 0;
    for seed in 0..32u64 {
        let shape = shapes::irreducible_shape(seed, 5, 2);
        let (mut module, _) = shapes::author_constant_seeded(&shape, 3);
        before += module.functions[0].blocks.len();
        crate::native::rewrites::prune_constant_cfg_module_if_changed(&mut module);
        after += module.functions.first().map_or(0, |f| f.blocks.len());
    }
    // Measured at 4131 -> 3772 when this was written: a modest fraction, because most blocks in a
    // depth-5 shape are inside a loop whose accumulator is not a constant. The floor is set below
    // the measurement, not at it.
    assert!(
        after * 20 <= before * 19 && after > 0,
        "constant folding took {before} blocks to {after}; it is not deciding what it should, or \
         it decided the whole function away"
    );
}

/// Executing a construction says it computes the right thing; it does not say a driver would
/// accept it. SPIR-V's structured control flow rules -- a block appears before every block it
/// dominates, a merge block is dominated by its header, a construct is left only at its own merge
/// or an enclosing loop's continue -- are all things a semantically correct rewrite can still get
/// wrong, and `spirv-val` is what knows them.
///
/// The authored shape is invalid on purpose (that is what selects construction), so only the
/// output is validated.
#[test]
fn constructed_shapes_are_valid_spirv() {
    let tmp = std::env::temp_dir().join("m2v_cfg_testkit_val");
    for depth in [3u32, 5] {
        for crossings in [0u32, 2] {
            for seed in 0..16u64 {
                let shape = shapes::irreducible_shape(seed, depth, crossings);
                let (mut module, _) = shapes::author_constant_seeded(&shape, 3);
                let _ = crate::native::rewrites::construct_cfg_functions_module(
                    &mut module,
                    &std::collections::HashSet::new(),
                );
                let bytes = assemble(&module);
                crate::tools::spirv_val_bytes(&bytes, &tmp).unwrap_or_else(|error| {
                    panic!(
                        "construction produced invalid SPIR-V: {error}\n{}",
                        shapes::describe(&shape)
                    )
                });
            }
        }
    }
}

/// A constructed module has to survive the trip through binary that a consumer puts it through.
/// Round-tripping it and comparing the disassembly catches an assembler and a parser that disagree
/// about an instruction's encoding, which neither validation nor execution of the in-memory module
/// would notice.
#[test]
fn constructed_shapes_survive_a_binary_round_trip_unchanged() {
    for depth in [3u32, 5] {
        for seed in 0..24u64 {
            let shape = shapes::irreducible_shape(seed, depth, 2);
            let (mut module, slot) = shapes::author_constant_seeded(&shape, 3);
            let _ = crate::native::rewrites::construct_cfg_functions_module(
                &mut module,
                &std::collections::HashSet::new(),
            );
            let expected = interp::run_module_to_global(&module, slot, STEP_LIMIT)
                .expect("constructed module runs");

            let reloaded = crate::spirv_module::load_bytes(assemble(&module))
                .unwrap_or_else(|error| panic!("reloading: {error:?}"));
            assert_eq!(
                reloaded.disassemble(),
                module.disassemble(),
                "a binary round trip changed this module:\n{}",
                shapes::describe(&shape)
            );
            assert_eq!(
                interp::run_module_to_global(&reloaded, slot, STEP_LIMIT).expect("reloaded runs"),
                expected,
                "a binary round trip changed what this module computes:\n{}",
                shapes::describe(&shape)
            );
        }
    }
}

fn assemble(module: &crate::spirv_module::Module) -> Vec<u8> {
    module
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
}
