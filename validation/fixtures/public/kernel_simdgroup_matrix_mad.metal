#include <metal_stdlib>
using namespace metal;

kernel void simdgroup_matrix_mad(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device const float *c [[buffer(2)]],
    device float *out [[buffer(3)]]) {
  simdgroup_float8x8 ma;
  simdgroup_float8x8 mb;
  simdgroup_float8x8 mc;
  simdgroup_float8x8 result;
  simdgroup_load(ma, a, 8);
  simdgroup_load(mb, b, 8);
  simdgroup_load(mc, c, 8);
  simdgroup_multiply_accumulate(result, ma, mb, mc);
  simdgroup_store(result, out, 8);
}
