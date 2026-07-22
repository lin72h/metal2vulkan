//! Byte-neutral responsibility split of the former monolith; see the parent module.

use crate::spirv_module::Instruction;
use crate::spirv_module::Operand;
use spirv::{Op, Word};
use std::collections::{HashMap, HashSet};

/// SSA values in `f` proven `!= 0` for every executing invocation. Seeded from a `NumWorkgroups`
/// load (each grid dimension is `>= 1`), grown through the value-preserving ops the dispatch-index
/// arithmetic uses: component extracts, integer conversions, and a product of two nonzero terms. A
/// nonzero MODULE constant is also a seed. (Narrowing conversions and wide products are treated as
/// nonzero-preserving under the kernel's own grid-fits-its-index-type contract — the same contract
/// the AGX kernel encodes by truncating the grid size to `ushort`; a grid that overflowed that type
/// would already miscompile in Apple's own code.)
pub(in crate::native) fn compute_nonzero(
    f: &crate::spirv_module::Function,
    consts: &HashMap<Word, i128>,
    numworkgroups: &HashSet<Word>,
) -> HashSet<Word> {
    let mut nz: HashSet<Word> = consts
        .iter()
        .filter(|(_, v)| **v != 0)
        .map(|(k, _)| *k)
        .collect();
    let is_nz = |nz: &HashSet<Word>, op: Option<&Operand>| -> bool {
        matches!(op, Some(Operand::IdRef(id)) if nz.contains(id))
    };
    loop {
        let mut changed = false;
        for b in &f.blocks {
            for inst in &b.instructions {
                let Some(rid) = inst.result_id else { continue };
                if nz.contains(&rid) {
                    continue;
                }
                let seed = match inst.class.opcode {
                    Op::Load => is_nz_load(inst, numworkgroups),
                    Op::CompositeExtract | Op::VectorExtractDynamic => {
                        is_nz(&nz, inst.operands.first())
                    }
                    Op::UConvert | Op::SConvert | Op::CopyObject | Op::Bitcast => {
                        is_nz(&nz, inst.operands.first())
                    }
                    Op::IMul => {
                        is_nz(&nz, inst.operands.first()) && is_nz(&nz, inst.operands.get(1))
                    }
                    _ => false,
                };
                if seed {
                    nz.insert(rid);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    nz
}

/// Whether a `Load` reads a `NumWorkgroups` builtin variable (directly).
pub(in crate::native) fn is_nz_load(inst: &Instruction, numworkgroups: &HashSet<Word>) -> bool {
    matches!(inst.operands.first(), Some(Operand::IdRef(p)) if numworkgroups.contains(p))
}

/// Resolve `id` to `(base, offset)` where `id == base + offset (mod 2^width)`, threading through
/// `IAdd`/`ISub` whose other operand is a known module constant. Non-affine ids are their own base
/// with offset 0.
pub(in crate::native) fn affine(
    id: Word,
    def: &HashMap<Word, &Instruction>,
    consts: &HashMap<Word, i128>,
    widths: &HashMap<Word, u32>,
) -> (Word, u128) {
    let mask = |w: u32| -> u128 {
        if w >= 128 {
            u128::MAX
        } else {
            (1u128 << w) - 1
        }
    };
    let mut base = id;
    let mut off: u128 = 0;
    let mut guard = 0;
    while guard < 64 {
        guard += 1;
        let Some(inst) = def.get(&base) else { break };
        let w = match inst.result_id.and_then(|r| widths.get(&r)) {
            Some(w) => *w,
            None => break,
        };
        let m = mask(w);
        let a = inst.operands.first();
        let b = inst.operands.get(1);
        let cst = |o: Option<&Operand>| -> Option<u128> {
            match o {
                Some(Operand::IdRef(c)) => consts.get(c).map(|v| (*v as u128) & m),
                _ => None,
            }
        };
        match inst.class.opcode {
            Op::IAdd => {
                if let (Some(Operand::IdRef(x)), Some(c)) = (a, cst(b)) {
                    off = off.wrapping_add(c) & m;
                    base = *x;
                } else if let (Some(c), Some(Operand::IdRef(x))) = (cst(a), b) {
                    off = off.wrapping_add(c) & m;
                    base = *x;
                } else {
                    break;
                }
            }
            Op::ISub => {
                if let (Some(Operand::IdRef(x)), Some(c)) = (a, cst(b)) {
                    off = off.wrapping_sub(c) & m;
                    base = *x;
                } else {
                    break;
                }
            }
            _ => break,
        }
    }
    (base, off)
}

/// Fold grid-stride early-return guards to `true`: an `OpUGreaterThan(X, Y)` where `Y == X - 1`
/// (affine, same base) and `X` is proven nonzero. `X > X-1` holds for every unsigned `X >= 1`, so
/// once the FC-derived work count folds the offset to exactly `-1` this guard is statically taken,
/// which lets branch-folding prune the guarded MXU compute nest. Returns `guard_id -> 1`.
pub(in crate::native) fn nonzero_self_minus_one_guards(
    f: &crate::spirv_module::Function,
    consts: &HashMap<Word, i128>,
    widths: &HashMap<Word, u32>,
    numworkgroups: &HashSet<Word>,
) -> HashMap<Word, i128> {
    if numworkgroups.is_empty() {
        return HashMap::new();
    }
    let nz = compute_nonzero(f, consts, numworkgroups);
    let mut def: HashMap<Word, &Instruction> = HashMap::new();
    for b in &f.blocks {
        for inst in &b.instructions {
            if let Some(r) = inst.result_id {
                def.insert(r, inst);
            }
        }
    }
    let mut out = HashMap::new();
    for b in &f.blocks {
        for inst in &b.instructions {
            if inst.class.opcode != Op::UGreaterThan {
                continue;
            }
            let Some(rid) = inst.result_id else { continue };
            let (Some(Operand::IdRef(x)), Some(Operand::IdRef(y))) =
                (inst.operands.first(), inst.operands.get(1))
            else {
                continue;
            };
            // X must be a proven-nonzero base (offset 0); Y must be exactly X - 1.
            let (bx, ox) = affine(*x, &def, consts, widths);
            let (by, oy) = affine(*y, &def, consts, widths);
            let w = widths.get(x).copied();
            let Some(w) = w else { continue };
            let mask = if w >= 128 {
                u128::MAX
            } else {
                (1u128 << w) - 1
            };
            if bx == by && ox == 0 && oy == mask && nz.contains(&bx) {
                out.insert(rid, 1);
            }
        }
    }
    out
}

/// Unsigned comparison kind for [`ucmp_fold`].
#[derive(Clone, Copy)]
pub(in crate::native) enum UCmp {
    Lt,
    Gt,
    Le,
    Ge,
}

/// Fold an unsigned integer comparison over lattice operands. Beyond the both-constant case, it
/// applies the exact unsigned bound `0 <= x` to fold a comparison with a known-0 operand REGARDLESS
/// of the other (even Bottom): `x < 0` and `0 > x` are always false; `0 <= x` and `x >= 0` are always
/// true. Values are interpreted unsigned (widths <= 64; the lattice stores width-masked non-negative
/// integers).
pub(in crate::native) fn ucmp_fold(x: Option<Lat>, y: Option<Lat>, kind: UCmp) -> Option<Lat> {
    let is_zero = |v: Option<Lat>| matches!(v, Some(Lat::Const(0)));
    match kind {
        UCmp::Lt if is_zero(y) => return Some(Lat::Const(0)), // x < 0 = false
        UCmp::Gt if is_zero(x) => return Some(Lat::Const(0)), // 0 > y = false
        UCmp::Le if is_zero(x) => return Some(Lat::Const(1)), // 0 <= y = true
        UCmp::Ge if is_zero(y) => return Some(Lat::Const(1)), // x >= 0 = true
        _ => {}
    }
    match (x, y) {
        (Some(Lat::Bottom), _) | (_, Some(Lat::Bottom)) => Some(Lat::Bottom),
        (Some(Lat::Const(a)), Some(Lat::Const(b))) => {
            let (a, b) = (a as u128, b as u128);
            let r = match kind {
                UCmp::Lt => a < b,
                UCmp::Gt => a > b,
                UCmp::Le => a <= b,
                UCmp::Ge => a >= b,
            };
            Some(Lat::Const(r as i128))
        }
        _ => None,
    }
}

/// A value in the constant lattice. Absent-from-the-map = TOP (optimistically undefined). This is
/// the classic SCCP lattice: TOP -> Const -> Bottom, monotone downward, so the fixpoint terminates.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::native) enum Lat {
    Const(i128),
    Bottom,
}

/// Meet two lattice values (TOP is represented by `None`).
pub(in crate::native) fn meet(a: Option<Lat>, b: Option<Lat>) -> Option<Lat> {
    match (a, b) {
        (None, x) | (x, None) => x, // TOP ∧ x = x
        (Some(Lat::Bottom), _) | (_, Some(Lat::Bottom)) => Some(Lat::Bottom),
        (Some(Lat::Const(x)), Some(Lat::Const(y))) => {
            Some(if x == y { Lat::Const(x) } else { Lat::Bottom })
        }
    }
}
