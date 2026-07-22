# AIR calls

This subsystem lowers residual stable AIR/LLVM intrinsic calls in the retained SPIR-V module:

- `dispatch_texture.rs` owns the single call-site dispatcher and texture/query/control intrinsics.
- `integer_simd.rs`, `float_imageblock.rs`, `reduce_bitops.rs`, and `matrix_shuffle.rs` own numeric,
  subgroup, imageblock, matrix, and shuffle families.
- `rawbyte_unary.rs`, `bfloat_glsl.rs`, and `conversions.rs` own raw-resource, unary math, bfloat,
  and conversion mechanisms.
- `images/` owns structural image resolution, coordinate/bounds construction, sampling, gather,
  query/offset, depth, read, and write helpers used by those call families.
