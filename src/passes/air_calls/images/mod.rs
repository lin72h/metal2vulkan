//! Texture image resolution, coordinate, sample, gather, read, and write helpers.

use super::*;

mod sample;
pub(in crate::passes) use sample::*;
mod gather;
pub(in crate::passes) use gather::*;
mod sample_depth;
pub(in crate::passes) use sample_depth::*;
mod write_read;
pub(in crate::passes) use write_read::*;
mod fetch_coord;
pub(in crate::passes) use fetch_coord::*;
mod query_offset;
pub(in crate::passes) use query_offset::*;
mod resolve;
pub(in crate::passes) use resolve::*;

#[cfg(test)]
mod coord_tests {
    //! Behavior pins for the texture-coordinate builders (S22). images.rs had no unit tests; these
    //! lock the observable shape of the sampled-image coordinate assembly BEFORE the coord builders
    //! are unified, so a byte-changing regression in the CompositeConstruct type/width/operand count
    //! is caught at the unit level (not only end-to-end via the execution gates).
    use super::*;
    use crate::spirv_module::Module;

    /// A minimal `Ctx` plus a fresh `%coord` value defined (in `out`) with the given vector type, and
    /// a `%layer` u32 constant defined in the module so `value_result_type` can resolve it.
    fn ctx_with_coord(vec_lanes: u32) -> (Ctx, Word, Word, Vec<Instruction>) {
        let mut ctx = Ctx::new(Module::new());
        let vty = ctx.ty_vecf(vec_lanes);
        let layer = ctx.const_uint(0);
        let coord = ctx.module.fresh_id();
        let out = vec![Instruction::new(Op::Undef, Some(vty), Some(coord), vec![])];
        (ctx, coord, layer, out)
    }

    #[test]
    fn build_sample_coord_non_arrayed_passes_coord_through() {
        for dim in [Dim::Dim1D, Dim::Dim2D, Dim::Dim3D, Dim::DimCube] {
            let (mut ctx, coord, _layer, mut out) = ctx_with_coord(2);
            let n_before = out.len();
            let got = build_sample_coord(&mut ctx, dim, false, coord, &[], &mut out).unwrap();
            assert_eq!(
                got, coord,
                "non-arrayed {dim:?} must return the coord unchanged"
            );
            assert_eq!(out.len(), n_before, "non-arrayed path emits nothing");
        }
    }

    #[test]
    fn build_sample_coord_2d_arrayed_appends_layer_as_vecf3() {
        let (mut ctx, coord, layer, mut out) = ctx_with_coord(2);
        let args = [0u32, 0, 0, layer];
        let combined =
            build_sample_coord(&mut ctx, Dim::Dim2D, true, coord, &args, &mut out).unwrap();
        let vf3 = ctx.ty_vecf(3);
        let last = out.last().unwrap();
        assert_eq!(last.class.opcode, Op::CompositeConstruct);
        assert_eq!(last.result_type, Some(vf3), "arrayed 2D coord is v3float");
        assert_eq!(last.result_id, Some(combined));
        // 2 extracted spatial floats + 1 float layer.
        assert_eq!(last.operands.len(), 3);
    }

    /// The gather-2D builder must produce the SAME coordinate shape as `build_sample_coord` at
    /// `dim = 2D` — the invariant the unification relies on.
    #[test]
    fn build_gather_coord_2d_matches_sample_coord_shape() {
        let (mut ctx_s, coord_s, layer_s, mut out_s) = ctx_with_coord(2);
        let args = [0u32, 0, 0, layer_s];
        let sample =
            build_sample_coord(&mut ctx_s, Dim::Dim2D, true, coord_s, &args, &mut out_s).unwrap();
        let sample_last = out_s.last().unwrap().clone();

        let (mut ctx_g, coord_g, layer_g, mut out_g) = ctx_with_coord(2);
        let gather =
            build_gather_coord_2d(&mut ctx_g, true, coord_g, Some(layer_g), &mut out_g).unwrap();
        let gather_last = out_g.last().unwrap();

        assert_eq!(sample_last.class.opcode, gather_last.class.opcode);
        assert_eq!(sample_last.result_type, gather_last.result_type);
        assert_eq!(sample_last.operands.len(), gather_last.operands.len());
        // Same instruction count emitted (extracts + layer convert + construct).
        assert_eq!(out_s.len(), out_g.len());
        let _ = (sample, gather);
    }
}
