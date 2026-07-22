//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;
use crate::spirv_module::Operand;
use spirv::{Op, Word};
use std::collections::HashMap;

/// SCCP-style iterative constant evaluation of a function's SSA values. Phis resolve only when every
/// incoming value is known and all incoming constants agree; unknown loop-carried backedges must not
/// be treated as neutral, or an induction variable's entry value can fold a real loop exit away.
/// Returns only the values that resolve to a definite constant. `consts` are module-scope constants;
/// `global_consts` are globals whose every load yields a known constant.
pub(in crate::native) fn forward_eval(
    f: &crate::spirv_module::Function,
    consts: &HashMap<Word, i128>,
    global_consts: &HashMap<Word, i128>,
    widths: &HashMap<Word, u32>,
    composites: &HashMap<Word, Vec<i128>>,
    vec_globals: &HashMap<Word, Vec<i128>>,
) -> HashMap<Word, i128> {
    let mut lat: HashMap<Word, Lat> = consts.iter().map(|(k, v)| (*k, Lat::Const(*v))).collect();
    // Known composite (vector) values by SSA id: seeded from module composite constants, then grown
    // as `OpLoad` of a known vector global and `OpCompositeConstruct` of known scalars resolve. A
    // composite value here is stable (only exact-constant sources), so it is written once and read by
    // `OpCompositeExtract` to yield a scalar constant.
    let mut comp: HashMap<Word, Vec<i128>> = composites.clone();
    let get = |lat: &HashMap<Word, Lat>, op: Option<&Operand>| -> Option<Lat> {
        match op {
            Some(Operand::IdRef(id)) => lat.get(id).copied(),
            _ => None,
        }
    };
    // Binary op over two integer-constant operands; TOP propagates (wait), Bottom poisons.
    let binop = |lat: &HashMap<Word, Lat>,
                 a: Option<&Operand>,
                 b: Option<&Operand>,
                 f: &dyn Fn(i128, i128) -> i128|
     -> Option<Lat> {
        match (get(lat, a), get(lat, b)) {
            (Some(Lat::Bottom), _) | (_, Some(Lat::Bottom)) => Some(Lat::Bottom),
            (Some(Lat::Const(x)), Some(Lat::Const(y))) => Some(Lat::Const(f(x, y))),
            _ => None, // some operand still TOP
        }
    };
    let mut changed = true;
    let mut guard = 0;
    while changed && guard < 256 {
        changed = false;
        guard += 1;
        for blk in &f.blocks {
            for inst in &blk.instructions {
                let Some(rid) = inst.result_id else { continue };
                // Grow the composite map: a load of a known-constant vector global, or a construct
                // over now-known scalar components, yields a known vector. New composite entries can
                // expose a foldable `OpCompositeExtract`, so mark progress to re-run the fixpoint.
                if let std::collections::hash_map::Entry::Vacant(e) = comp.entry(rid) {
                    let cv: Option<Vec<i128>> = match inst.class.opcode {
                        Op::Load | Op::CopyObject => match inst.operands.first() {
                            Some(Operand::IdRef(p)) => vec_globals.get(p).cloned(),
                            _ => None,
                        },
                        // Only immutable MODULE scalar constants are read here (never a flow value
                        // that could later drop to Bottom), so a committed composite is sound and
                        // stable — no monotonicity hazard.
                        Op::CompositeConstruct => inst
                            .operands
                            .iter()
                            .map(|op| match op {
                                Operand::IdRef(c) => consts.get(c).copied(),
                                _ => None,
                            })
                            .collect(),
                        _ => None,
                    };
                    if let Some(cv) = cv {
                        e.insert(cv);
                        changed = true;
                    }
                }
                let new: Option<Lat> = match inst.class.opcode {
                    // Extract a scalar component from a known constant vector at a constant index.
                    Op::CompositeExtract => match (inst.operands.first(), inst.operands.get(1)) {
                        (Some(Operand::IdRef(src)), Some(Operand::LiteralBit32(idx))) => {
                            match comp.get(src) {
                                Some(v) => v.get(*idx as usize).map(|c| Lat::Const(*c)),
                                None => Some(Lat::Bottom),
                            }
                        }
                        _ => Some(Lat::Bottom),
                    },
                    Op::Phi => {
                        // A phi is foldable only when every incoming value is already known and the
                        // incoming constants agree. Treating TOP/unknown incoming edges as neutral is
                        // unsafe for loop-carried induction variables: the entry value can otherwise
                        // be mistaken for the fixed-point value and fold a real loop exit away.
                        let mut acc: Option<Lat> = None;
                        let mut i = 0;
                        while i < inst.operands.len() {
                            if let Some(Operand::IdRef(v)) = inst.operands.get(i) {
                                let Some(value) = lat.get(v).copied() else {
                                    acc = None;
                                    break;
                                };
                                acc = meet(acc, Some(value));
                            }
                            i += 2;
                        }
                        acc
                    }
                    Op::Load => match inst.operands.first() {
                        Some(Operand::IdRef(p)) => global_consts.get(p).copied().map(Lat::Const),
                        _ => Some(Lat::Bottom),
                    },
                    Op::CopyObject => get(&lat, inst.operands.first()),
                    // Conversions/bitcasts: forward ONLY a zero operand (0 maps to 0 under every
                    // integer widen/narrow/reinterpret), keeping the fold sound without tracking bit
                    // widths. The FC-predicate path is all zeros, so this suffices.
                    Op::UConvert | Op::SConvert | Op::Bitcast => {
                        match get(&lat, inst.operands.first()) {
                            Some(Lat::Const(0)) => Some(Lat::Const(0)),
                            Some(Lat::Bottom) => Some(Lat::Bottom),
                            _ => None,
                        }
                    }
                    Op::IEqual => binop(
                        &lat,
                        inst.operands.first(),
                        inst.operands.get(1),
                        &|a, b| (a == b) as i128,
                    ),
                    Op::INotEqual => binop(
                        &lat,
                        inst.operands.first(),
                        inst.operands.get(1),
                        &|a, b| (a != b) as i128,
                    ),
                    // Unsigned comparisons: fold when both operands are known, AND — soundly, via the
                    // unsigned bound `x >= 0` — when just one operand is the constant 0 (an FC-derived
                    // dimension count that folded to 0), regardless of the other. `x < 0` / `0 > x` are
                    // FALSE for every unsigned `x`; `0 <= x` / `x >= 0` are TRUE. This folds the
                    // `tile < (W*H = 0)` grid-bounds guard that skips an FC-zero-work MXU compute nest.
                    Op::ULessThan => ucmp_fold(
                        get(&lat, inst.operands.first()),
                        get(&lat, inst.operands.get(1)),
                        UCmp::Lt,
                    ),
                    Op::UGreaterThan => ucmp_fold(
                        get(&lat, inst.operands.first()),
                        get(&lat, inst.operands.get(1)),
                        UCmp::Gt,
                    ),
                    Op::ULessThanEqual => ucmp_fold(
                        get(&lat, inst.operands.first()),
                        get(&lat, inst.operands.get(1)),
                        UCmp::Le,
                    ),
                    Op::UGreaterThanEqual => ucmp_fold(
                        get(&lat, inst.operands.first()),
                        get(&lat, inst.operands.get(1)),
                        UCmp::Ge,
                    ),
                    Op::LogicalNot => match get(&lat, inst.operands.first()) {
                        Some(Lat::Const(a)) => Some(Lat::Const((a == 0) as i128)),
                        Some(Lat::Bottom) => Some(Lat::Bottom),
                        None => None,
                    },
                    Op::LogicalAnd => binop(
                        &lat,
                        inst.operands.first(),
                        inst.operands.get(1),
                        &|a, b| ((a != 0) && (b != 0)) as i128,
                    ),
                    Op::LogicalOr => binop(
                        &lat,
                        inst.operands.first(),
                        inst.operands.get(1),
                        &|a, b| ((a != 0) || (b != 0)) as i128,
                    ),
                    // Pure integer bitwise ops + logical shift-RIGHT over two known constants —
                    // standard SCCP modeling, restricted to the WIDTH-INDEPENDENT operations. The
                    // `air.normalize_function_constant_predicate` lowering packs several boolean
                    // function-constant flags into one word and extracts each with a
                    // shift-right+mask+xor chain (`(fc >> k) & 1`, `^ 1`), so folding the FC
                    // predicate to its disabled default requires these. Operands are exact unsigned
                    // values (module constants, or 0 from a narrowing conversion — the UConvert arm
                    // only forwards 0), so AND/OR/XOR and a LOGICAL right shift are exact regardless
                    // of the SPIR-V type width (the result's bits never exceed the operands' true
                    // bits). Shift-LEFT is deliberately NOT modeled: it is width-sensitive
                    // (`8u8 << 5` truncates to 0 in SPIR-V but not in i128), so folding it could
                    // mis-decide a downstream branch and `forward_eval` tracks no width to mask it.
                    Op::BitwiseAnd => binop(
                        &lat,
                        inst.operands.first(),
                        inst.operands.get(1),
                        &|a, b| a & b,
                    ),
                    Op::BitwiseOr => binop(
                        &lat,
                        inst.operands.first(),
                        inst.operands.get(1),
                        &|a, b| a | b,
                    ),
                    Op::BitwiseXor => binop(
                        &lat,
                        inst.operands.first(),
                        inst.operands.get(1),
                        &|a, b| a ^ b,
                    ),
                    Op::ShiftRightLogical => binop(
                        &lat,
                        inst.operands.first(),
                        inst.operands.get(1),
                        &|a, b| {
                            if (0..128).contains(&b) {
                                ((a as u128) >> b) as i128
                            } else {
                                0
                            }
                        },
                    ),
                    // WIDTH-SENSITIVE integer arithmetic: fold ONLY when the result type's bit width
                    // is known, masking the two's-complement result to that width so it matches
                    // SPIR-V's modular semantics exactly (a `ushort n_requ = (W*H + 15) >> 4` dead-
                    // dimension guard needs IMul/IAdd; the shift-right above finishes it). Unknown
                    // width ⇒ not folded (stays TOP), never mis-folded.
                    Op::IAdd | Op::IMul | Op::ISub => match widths.get(&rid) {
                        Some(&w) => {
                            let mask = if w >= 128 {
                                u128::MAX
                            } else {
                                (1u128 << w) - 1
                            };
                            let op = |f: &dyn Fn(u128, u128) -> u128| -> Option<Lat> {
                                match (
                                    get(&lat, inst.operands.first()),
                                    get(&lat, inst.operands.get(1)),
                                ) {
                                    (Some(Lat::Bottom), _) | (_, Some(Lat::Bottom)) => {
                                        Some(Lat::Bottom)
                                    }
                                    (Some(Lat::Const(x)), Some(Lat::Const(y))) => {
                                        Some(Lat::Const((f(x as u128, y as u128) & mask) as i128))
                                    }
                                    _ => None,
                                }
                            };
                            match inst.class.opcode {
                                Op::IAdd => op(&|a, b| a.wrapping_add(b)),
                                Op::IMul => op(&|a, b| a.wrapping_mul(b)),
                                _ => op(&|a, b| a.wrapping_sub(b)),
                            }
                        }
                        None => None,
                    },
                    Op::Select => match get(&lat, inst.operands.first()) {
                        Some(Lat::Const(c)) => {
                            let arm = if c != 0 {
                                inst.operands.get(1)
                            } else {
                                inst.operands.get(2)
                            };
                            get(&lat, arm)
                        }
                        Some(Lat::Bottom) => Some(Lat::Bottom),
                        None => None,
                    },
                    _ => Some(Lat::Bottom), // unmodeled op: its result is not a tracked constant
                };
                // Monotone update: only move downward (TOP -> Const -> Bottom).
                if let Some(n) = new {
                    let cur = lat.get(&rid).copied();
                    let merged = match (cur, n) {
                        (None, n) => Some(n),
                        (Some(Lat::Bottom), _) => Some(Lat::Bottom),
                        (Some(Lat::Const(_)), Lat::Bottom) => Some(Lat::Bottom),
                        (Some(Lat::Const(a)), Lat::Const(b)) if a == b => Some(Lat::Const(a)),
                        (Some(Lat::Const(_)), Lat::Const(_)) => Some(Lat::Bottom),
                    };
                    if merged != cur {
                        lat.insert(rid, merged.unwrap());
                        changed = true;
                    }
                }
            }
        }
    }
    lat.into_iter()
        .filter_map(|(k, v)| match v {
            Lat::Const(c) => Some((k, c)),
            Lat::Bottom => None,
        })
        .collect()
}
