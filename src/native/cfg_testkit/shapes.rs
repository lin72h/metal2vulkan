//! Generate reducible-but-unstructured control-flow graphs, deterministically from a seed.
//!
//! The shapes that break a structurizer are not the ones anyone writes by hand. They are the ones a
//! real front end produces: a `break` out of three nested constructs at once, a `continue` to a
//! grandparent loop, a conditional whose arms rejoin somewhere other than where the structure says
//! they should. Each is easy to describe and tedious to author, so the authored set stays small and
//! the interesting combinations go untested.
//!
//! A generator fixes the ratio. A region tree (sequence / conditional / loop) is grown from a
//! seeded PRNG and then perforated with escape edges that jump to an enclosing region's exit or an
//! enclosing loop's header — every edge either forward or a back-edge to a dominating header, so
//! the result is always reducible, and almost never structured.
//!
//! Two properties make the output usable as a differential input:
//!
//! - **It terminates.** One accumulator threads the whole function, every block adds a positive
//!   amount to it, and every loop's back-edge test is `accumulator < bound`. A monotonically
//!   increasing accumulator cannot cycle, so no generated shape can hang the interpreter — a step
//!   limit being hit therefore means the *construction* stopped terminating.
//! - **Its value flow is real.** That accumulator is live across every construct boundary and
//!   `OpPhi`-merged at every join, so a rewrite that mis-selects a phi incoming or loses an edge
//!   changes the returned value instead of quietly producing an equally plausible one.

use super::build::CfgBuilder;
use crate::spirv_module::Module;

/// Upper bound on the accumulator, and so on every generated loop's trip count.
const ACCUMULATOR_BOUND: u32 = 4096;

/// A generated block's terminator, before it is given labels.
#[derive(Clone, Debug)]
enum Term {
    Branch(usize),
    /// `condition ? on_true : on_false`, over the accumulator live at this block.
    Conditional(Cond, usize, usize),
    /// `switch (accumulator & mask)`, with `arms[k]` for selector `k` and `default` for the rest.
    /// A switch construct is not a conditional with more arms: SPIR-V lets a nested construct
    /// break out to an enclosing switch and not to an enclosing selection, so the two take
    /// different paths through the emitter.
    Multiway {
        mask: u32,
        arms: Vec<usize>,
        default: usize,
    },
    Return,
}

/// A branch condition, always a function of the accumulator so control flow depends on data.
#[derive(Clone, Copy, Debug)]
enum Cond {
    /// `(accumulator & mask) == 0`
    MaskIsZero(u32),
    /// `accumulator < bound` — the only condition a loop back-edge test uses, so that a
    /// monotonically increasing accumulator guarantees the loop exits.
    LessThan(u32),
}

/// A tiny deterministic PRNG. Seeded shapes have to be reproducible from the seed alone, which
/// rules out anything ambient.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // SplitMix64.
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u32) -> u32 {
        (self.next() % u64::from(bound)) as u32
    }

    /// True with probability `numerator / 16`.
    fn chance(&mut self, numerator: u32) -> bool {
        self.below(16) < numerator
    }
}

struct Generator {
    rng: Rng,
    terminators: Vec<Option<Term>>,
    /// Blocks already given a terminator, as candidates for an irreducible cross edge.
    settled: Vec<usize>,
    /// Headers of the loops currently enclosing the region being emitted, outermost first.
    loop_headers: Vec<usize>,
    /// Exit blocks of the regions currently enclosing the one being emitted, outermost first.
    region_exits: Vec<usize>,
    /// How many escape edges the shape has taken so far, so a shape stays finite.
    escapes: u32,
    escape_budget: u32,
    /// How many edges into the interior of an already-written region are still allowed. Each one
    /// gives some loop a second entry, so a graph with any of them is irreducible and only the
    /// state-machine constructor can express it.
    cross_budget: u32,
}

impl Generator {
    fn allocate(&mut self) -> usize {
        self.terminators.push(None);
        self.terminators.len() - 1
    }

    fn set(&mut self, block: usize, term: Term) {
        assert!(
            self.terminators[block].is_none(),
            "block {block} terminated twice"
        );
        self.terminators[block] = Some(term);
        self.settled.push(block);
    }

    /// An edge into the interior of a region already written, which is what makes the graph
    /// irreducible: the target keeps its original predecessors and gains one that does not go
    /// through whatever header used to dominate it.
    ///
    /// The edge is guarded by the same monotonic `accumulator < bound` test the loops use, never by
    /// a mask, so it stops being taken once the accumulator has grown past the bound. Without that,
    /// a cross edge could close a cycle containing no bounded test at all and the shape would not
    /// terminate.
    fn cross_edge(&mut self, from: usize) -> Option<(Cond, usize)> {
        if self.cross_budget == 0 {
            return None;
        }
        // Never the entry block: it has no `OpPhi` to merge a second arrival into, because nothing
        // precedes the first one. Never an enclosing header or exit either — those already
        // dominate this block, so an edge to one is an ordinary back edge or break and leaves the
        // graph reducible, which is not what this is for.
        let candidates = self
            .settled
            .iter()
            .copied()
            .filter(|block| {
                *block != 0
                    && *block != from
                    && !self.loop_headers.contains(block)
                    && !self.region_exits.contains(block)
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return None;
        }
        self.cross_budget -= 1;
        let pick = self.rng.below(candidates.len() as u32) as usize;
        let bound = ACCUMULATOR_BOUND - self.rng.below(64);
        Some((Cond::LessThan(bound), candidates[pick]))
    }

    /// A target an escape edge from inside the current region may jump to: an enclosing region's
    /// exit (a multi-level break) or an enclosing loop's header (a continue).
    fn escape_target(&mut self) -> Option<usize> {
        if self.escapes >= self.escape_budget {
            return None;
        }
        let candidates = self
            .region_exits
            .iter()
            .chain(&self.loop_headers)
            .copied()
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return None;
        }
        self.escapes += 1;
        let pick = self.rng.below(candidates.len() as u32) as usize;
        Some(candidates[pick])
    }

    /// Terminate `block` with a branch to `next`, or — when a target is available and the coin
    /// falls that way — with a conditional that leaves the current construct entirely or jumps
    /// into the interior of one already written.
    ///
    /// The cross-edge coin is flipped whether or not any crossings are allowed, so a seed names
    /// the same underlying shape with and without them.
    fn flow(&mut self, block: usize, next: usize) {
        let crossing = self.rng.chance(4);
        if crossing {
            if let Some((cond, target)) = self.cross_edge(block) {
                self.set(block, Term::Conditional(cond, target, next));
                return;
            }
        }
        if self.rng.chance(6) {
            if let Some(target) = self.escape_target() {
                let mask = 1 << self.rng.below(5);
                self.set(
                    block,
                    Term::Conditional(Cond::MaskIsZero(mask), target, next),
                );
                return;
            }
        }
        self.set(block, Term::Branch(next));
    }

    /// Emit a region that starts at `entry` (already allocated, not yet terminated) and leaves to
    /// `exit`.
    fn region(&mut self, entry: usize, exit: usize, depth: u32) {
        if depth == 0 {
            self.flow(entry, exit);
            return;
        }
        match self.rng.below(4) {
            0 => self.sequence(entry, exit, depth),
            1 => self.conditional(entry, exit, depth),
            2 => self.multiway(entry, exit, depth),
            _ => self.loop_region(entry, exit, depth),
        }
    }

    fn multiway(&mut self, entry: usize, exit: usize, depth: u32) {
        let arms = (0..3).map(|_| self.allocate()).collect::<Vec<_>>();
        let default = self.allocate();
        self.set(
            entry,
            Term::Multiway {
                mask: 3,
                arms: arms.clone(),
                default,
            },
        );
        self.region_exits.push(exit);
        for arm in arms.iter().chain(std::iter::once(&default)) {
            self.region(*arm, exit, depth - 1);
        }
        self.region_exits.pop();
    }

    fn sequence(&mut self, entry: usize, exit: usize, depth: u32) {
        let length = 1 + self.rng.below(3) as usize;
        let mut current = entry;
        for step in 0..length {
            let next = if step + 1 == length {
                exit
            } else {
                self.allocate()
            };
            self.region_exits.push(next);
            self.region(current, next, depth - 1);
            self.region_exits.pop();
            current = next;
        }
    }

    fn conditional(&mut self, entry: usize, exit: usize, depth: u32) {
        let on_true = self.allocate();
        let on_false = self.allocate();
        let mask = 1 << self.rng.below(5);
        self.set(
            entry,
            Term::Conditional(Cond::MaskIsZero(mask), on_true, on_false),
        );
        self.region_exits.push(exit);
        self.region(on_true, exit, depth - 1);
        self.region(on_false, exit, depth - 1);
        self.region_exits.pop();
    }

    fn loop_region(&mut self, header: usize, exit: usize, depth: u32) {
        let body = self.allocate();
        let latch = self.allocate();
        let bound = ACCUMULATOR_BOUND - self.rng.below(64);
        self.set(header, Term::Conditional(Cond::LessThan(bound), body, exit));
        self.loop_headers.push(header);
        self.region_exits.push(latch);
        self.region(body, latch, depth - 1);
        self.region_exits.pop();
        self.loop_headers.pop();
        // The latch closes the back edge unconditionally: the header owns the exit test, which is
        // what keeps the loop's trip count tied to the monotonic accumulator.
        self.set(latch, Term::Branch(header));
    }
}

/// A generated shape: the abstract graph plus the seed that produced it.
pub(in crate::native) struct Shape {
    seed: u64,
    terminators: Vec<Term>,
}

impl Shape {
    pub(in crate::native) fn blocks(&self) -> usize {
        self.terminators.len()
    }

    /// How many blocks end in a branch with more than one target — a rough measure of how much
    /// branching the shape has for a structurizer to get wrong.
    pub(in crate::native) fn branching(&self) -> usize {
        self.terminators
            .iter()
            .filter(|term| matches!(term, Term::Conditional(..) | Term::Multiway { .. }))
            .count()
    }
}

/// Grow one reducible shape from `seed`. `depth` bounds the region tree, so it bounds the block
/// count.
pub(in crate::native) fn shape(seed: u64, depth: u32) -> Shape {
    grow(seed, depth, 0)
}

/// Grow one shape from `seed` with up to `crossings` edges into the interior of an already-written
/// region.
///
/// Any such edge makes the graph irreducible, which the nesting structurizer declines by contract,
/// so this is how the state-machine constructor gets exercised on its own territory rather than
/// only on what nesting hands back.
pub(in crate::native) fn irreducible_shape(seed: u64, depth: u32, crossings: u32) -> Shape {
    grow(seed, depth, crossings)
}

fn grow(seed: u64, depth: u32, crossings: u32) -> Shape {
    let mut generator = Generator {
        rng: Rng(seed),
        terminators: Vec::new(),
        settled: Vec::new(),
        loop_headers: Vec::new(),
        region_exits: Vec::new(),
        escapes: 0,
        escape_budget: 24,
        cross_budget: crossings,
    };
    // The function entry is its own block, branching into the region tree. It cannot be a region
    // entry: the top-level region may be a loop, and a loop header needs a predecessor other than
    // its latch for its `OpPhi` to merge anything. An entry block that is also a loop header would
    // silently drop the back edge's value and never terminate.
    let entry = generator.allocate();
    let exit = generator.allocate();
    let start = generator.allocate();
    generator.set(entry, Term::Branch(start));
    generator.region(start, exit, depth);
    generator.set(exit, Term::Return);
    let terminators = generator
        .terminators
        .into_iter()
        .map(|term| term.expect("every allocated block is terminated"))
        .collect();
    Shape { seed, terminators }
}

/// Author `shape` as a `uint (uint)` SPIR-V function.
///
/// Every block adds a positive, block-specific amount to the accumulator threaded through the
/// function; joins merge it with `OpPhi`; the exit returns it. The single argument seeds it, so
/// different arguments take different paths through the same graph.
pub(in crate::native) fn author(shape: &Shape) -> Module {
    let mut builder = CfgBuilder::new(1);
    let name = |block: usize| format!("b{block}");

    let predecessors = predecessors(&shape.terminators);
    // The accumulator each block leaves with. A loop header's `OpPhi` names the value its latch
    // leaves with, and the latch is authored after the header, so these ids are reserved up front
    // and defined as each block is authored.
    let leaving = (0..shape.terminators.len())
        .map(|_| builder.reserve_value())
        .collect::<Vec<_>>();
    for (block, term) in shape.terminators.iter().enumerate() {
        builder.block(&name(block));
        let entering = if block == 0 {
            assert!(
                predecessors[0].is_empty(),
                "the entry block has predecessors, so its accumulator would ignore them"
            );
            builder.parameter(0)
        } else {
            let incoming = predecessors[block]
                .iter()
                .map(|predecessor| (leaving[*predecessor], name(*predecessor)))
                .collect::<Vec<_>>();
            assert!(
                !incoming.is_empty(),
                "block {block} is unreachable, so the shape is not connected"
            );
            if incoming.len() == 1 {
                incoming[0].0
            } else {
                let incoming = incoming
                    .iter()
                    .map(|(value, predecessor)| (*value, predecessor.as_str()))
                    .collect::<Vec<_>>();
                builder.phi(&incoming)
            }
        };
        // A positive, block-specific step: monotonic, so every loop terminates, and distinct, so a
        // path taken in the wrong order shows up in the result.
        let step = builder.constant(block as u32 + 1);
        let value = leaving[block];
        builder.add_into(value, entering, step);

        match term {
            Term::Branch(target) => builder.branch(&name(*target)),
            Term::Conditional(cond, on_true, on_false) => {
                let condition = match cond {
                    Cond::MaskIsZero(mask) => {
                        let mask = builder.constant(*mask);
                        let masked = builder.bitwise_and(value, mask);
                        let zero = builder.constant(0);
                        builder.equal(masked, zero)
                    }
                    Cond::LessThan(bound) => {
                        let bound = builder.constant(*bound);
                        builder.less_than(value, bound)
                    }
                };
                builder.branch_conditional(condition, &name(*on_true), &name(*on_false));
            }
            Term::Multiway {
                mask,
                arms,
                default,
            } => {
                let mask = builder.constant(*mask);
                let selector = builder.bitwise_and(value, mask);
                let cases = arms
                    .iter()
                    .enumerate()
                    .map(|(literal, arm)| (literal as u32, name(*arm)))
                    .collect::<Vec<_>>();
                let cases = cases
                    .iter()
                    .map(|(literal, target)| (*literal, target.as_str()))
                    .collect::<Vec<_>>();
                builder.switch(selector, &name(*default), &cases);
            }
            Term::Return => builder.return_value(value),
        }
    }
    builder.finish()
}

/// A block's predecessors, in the order the blocks appear, so authored `OpPhi` operand order is a
/// function of the shape alone.
fn predecessors(terminators: &[Term]) -> Vec<Vec<usize>> {
    let mut predecessors = vec![Vec::new(); terminators.len()];
    for (block, term) in terminators.iter().enumerate() {
        let mut edge = |target: usize| {
            if !predecessors[target].contains(&block) {
                predecessors[target].push(block);
            }
        };
        match term {
            Term::Branch(target) => edge(*target),
            Term::Conditional(_, on_true, on_false) => {
                edge(*on_true);
                edge(*on_false);
            }
            Term::Multiway { arms, default, .. } => {
                for arm in arms {
                    edge(*arm);
                }
                edge(*default);
            }
            Term::Return => {}
        }
    }
    predecessors
}

/// A one-line-per-block rendering of the abstract graph, for diagnosing a generator or
/// structurizer failure without disassembling the module.
pub(in crate::native) fn describe(shape: &Shape) -> String {
    let header = format!(
        "seed {} ({} blocks, {} branching)",
        shape.seed,
        shape.blocks(),
        shape.branching()
    );
    let blocks = shape
        .terminators
        .iter()
        .enumerate()
        .map(|(block, term)| match term {
            Term::Branch(target) => format!("b{block} -> b{target}"),
            Term::Conditional(cond, on_true, on_false) => {
                format!("b{block} -> {cond:?} ? b{on_true} : b{on_false}")
            }
            Term::Multiway {
                mask,
                arms,
                default,
            } => {
                let arms = arms
                    .iter()
                    .enumerate()
                    .map(|(literal, arm)| format!("{literal} => b{arm}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("b{block} -> switch (acc & {mask}) {{ {arms}, _ => b{default} }}")
            }
            Term::Return => format!("b{block} return"),
        });
    std::iter::once(header)
        .chain(blocks)
        .collect::<Vec<_>>()
        .join("\n")
}
