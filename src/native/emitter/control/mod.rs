use super::*;
use crate::native::cfg::graph::{
    bit_is_set, clear_unused_bits, set_bit, spirv_block_successors as cloned_block_successors,
    spirv_label_dominates as label_dominates,
};

mod loop_continue;
mod merge_helpers;
mod merges;
pub(in crate::native::emitter) use merge_helpers::*;

#[cfg(test)]
mod tests {
    /// `emit_branch` now derives its target labels from `tir::parse_terminator` (the typed-IR parser),
    /// which must strip trailing `, !llvm.loop !N` metadata off a branch target — the job the retired
    /// `branch_label_without_metadata` used to do in this file.
    #[test]
    fn typed_terminator_strips_loop_metadata_off_branch_target() {
        use crate::native::tir::{parse_terminator, TirTerminator};
        assert_eq!(
            parse_terminator("br label %59, !llvm.loop !47"),
            Some(TirTerminator::Br("%59".to_string()))
        );
        assert_eq!(
            parse_terminator("br i1 %c, label %t, label %f, !llvm.loop !9"),
            Some(TirTerminator::BrCond {
                cond: "%c".to_string(),
                t: "%t".to_string(),
                f: "%f".to_string(),
            })
        );
    }
}
