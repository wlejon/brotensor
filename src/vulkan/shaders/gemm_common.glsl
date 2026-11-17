// Shared by the GEMM kernels (gemm_cm.comp, gemm_simt.comp, gemv.comp): the
// push block and (gemm_act.glsl) the fused-epilogue activation.
//
// C[z](M, N) = op(A[z]) * op(B[z]), z < batch, element offsets z * s{a,b,c}
// (a zero stride broadcasts that operand). Layouts, chosen by the kernel's
// layout constants:
//   A: (M, K) row-major with row stride lda, or with TA set, stored (K, M)
//   B: (N, K) row-major with stride ldb (the linear-layer "NT" form), or
//      with NB set, stored (K, N)
// Epilogue on r = acc + bias[n] (bias may be absent), r = act(r):
//   EPI_STORE   C = r
//   EPI_ACCUM   C += r
//   EPI_GEGLU   C[:, j] = r[:, 2j] * gelu_exact(r[:, 2j+1])        (N even)
//   EPI_SWIGLU  C[:, j] = silu(r[:, j]) * r[:, half + j]            (rows of B
//               [0, half) are the gate, [half, 2 half) the up projection)

#include "op_codes.h"

layout(push_constant) uniform PC {
    uint64_t a, b, c, bias;
    uint64_t scale;        // gemm_cm with a quantised B (QB): INT8 / FP8 per-row scales
    uint64_t ascale;       // gemm_cm with SCALE_A: int exponent per (z, row of A), gemm_rowscale.comp
    uint m, n, k;          // n: columns of r (rows of B), 2 * half for the GLUs
    uint lda, ldb, ldc;    // gemm_cm with QB: ldb is B's row pitch in bytes
    uint sa, sb, sc;       // batch strides in elements
    uint half_n;           // EPI_SWIGLU: offset of the up rows (= output width)
    uint tm0, tn0;         // gemm_cm: tile origin of this dispatch (edge strips)
} pc;

#include "gemm_act.glsl"

uint64_t batch_base(uint64_t base, uint stride, uint z, const uint esize) {
    return base + uint64_t(z) * uint64_t(stride) * uint64_t(esize);
}
