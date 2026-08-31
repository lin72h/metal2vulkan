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
