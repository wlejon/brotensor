#pragma once

// Short-M 16-bit linear on the tensor cores, for 9..64 rows:
//
//   Y(M, N) = epilogue( X(M, K) · W(N, K)^T + bias )
//
// The regime of a transformer scoring a few dozen rows against its whole
// weight set (a flow-matching step over a latent window, a speculative verify,
// a short prefill). The product is weight-bandwidth bound there, so the
// kernel is built around streaming W once at full rate: W (the mma's A
// operand) and X (its B operand) flow through a 4-deep cp.async ring of
// 64-wide K chunks, filled eight lanes to a 128-byte row, and every warp
// reads its fragments with one 16-byte load per operand per 32-wide step
// (the K order permuted identically in both operands). A CTA covers 128
// output columns (8 warps x 16); when that leaves the SMs idle, K is split
// across CTAs into FP32 partials that a second pass sums in split order (so
// the result is deterministic) and finishes with the epilogue. One layer of
// a 2048-wide model at 52 rows: 140 us vs 211 us on the tile GEMM (4090,
// graph replay; tests/bench_skinny_linear.cpp).
//
// Epilogues (as linear_forward_batched_ex): 0 store act(r), 1 accumulate
// Y += act(r), 2 GeGLU (pairwise interleaved rows: Y[j] = r[2j] *
// gelu_exact(r[2j+1])), 3 SwiGLU (halves: Y[j] = silu(r[j]) * r[N/2 + j]),
// all in FP32 from the sum plus bias.
//
// launch() returns false and issues nothing when the shape or operands cannot
// take this path (M outside 9..64, K % 8, unaligned X / W, < sm_80, a gated
// epilogue with odd N); the caller falls back.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace brotensor::detail::cuda::gemm_skinny {

// FP32 elements of workspace a split launch of this shape wants (0 = unsplit).
std::size_t workspace_floats(int M, int N, int K, int epi);

bool launch(const __half* X, const __half* W, __half* Y, int M, int N, int K, const __half* bias, int act, int epi,
            float* ws, std::size_t ws_floats, cudaStream_t stream);
bool launch(const __nv_bfloat16* X, const __nv_bfloat16* W, __nv_bfloat16* Y, int M, int N, int K,
            const __nv_bfloat16* bias, int act, int epi, float* ws, std::size_t ws_floats, cudaStream_t stream);

}  // namespace brotensor::detail::cuda::gemm_skinny
