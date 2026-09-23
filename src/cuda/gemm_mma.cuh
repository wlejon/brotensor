#pragma once

// Tensor-core GEMM (mma.sync m16n8k16, FP32 accumulate, cp.async multistage
// pipeline, ldmatrix off a swizzled shared layout) for the 16-bit linear:
//
//   C(M, N) = epilogue( A(M, K) · W(N, K)^T + bias )
//
// Epilogues:
//   kStore      C  = act(r)
//   kAccumulate C += act(r)            (residual add fused into the store)
//   kGeglu      C(M, N/2)[j] = r[2j] * gelu_exact(r[2j+1])   (act ignored;
//               W's rows interleave the two GeGLU halves pairwise)
//
// Short-M products leave most SMs idle (each output tile is one SM's worth of
// tensor-core time), so given an FP32 workspace launch() splits K across
// CTAs, writes per-split partials, and a second kernel reduces them in a
// fixed order (deterministic) and applies the epilogue.
//
// launch() returns false and issues nothing when the device (< sm_80) or the
// operands (K or N not a multiple of 8, unaligned pointers) cannot take this
// path; callers fall back to the WMMA kernel.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace brotensor::detail::cuda::mma_gemm {

enum Epilogue : int { kStore = 0, kAccumulate = 1, kGeglu = 2 };

// Accumulation: kAccF32 = FP32 throughout. kAccHybrid (FP16 only) = each
// k16 mma step accumulates in FP16 at the doubled consumer-GPU tensor rate and
// is folded into FP32, so the error is a 16-term FP16 sum's, not K's.
// (Whole-K FP16 accumulation was tried: ~1.25x faster at large M but it moves
// Laya logits by 0.5, past parity — so no pure-FP16 mode.)
enum AccMode : int { kAccF32 = 0, kAccHybrid = 1 };

// FP32 elements of workspace a split-K launch of this shape wants (0 = the
// product runs unsplit).
std::size_t workspace_floats(int M, int N, int K);

bool launch(const __half* A, const __half* W, __half* C, int M, int N, int K, const __half* bias, int act,
            int epi, float* ws, std::size_t ws_floats, cudaStream_t stream, int acc_mode = kAccF32);
bool launch(const __nv_bfloat16* A, const __nv_bfloat16* W, __nv_bfloat16* C, int M, int N, int K,
            const __nv_bfloat16* bias, int act, int epi, float* ws, std::size_t ws_floats,
            cudaStream_t stream, int acc_mode = kAccF32);

// C(M, N) = A(M, K) · W(N, K)^T with the FP32 accumulators stored as FP32 —
// for a product whose consumer needs more than 16 bits (attention scores ahead
// of a softmax). No epilogue, never split. Same preconditions and false return
// as launch().
bool launch_f32out(const __half* A, const __half* W, float* C, int M, int N, int K, cudaStream_t stream);
bool launch_f32out(const __nv_bfloat16* A, const __nv_bfloat16* W, float* C, int M, int N, int K,
                   cudaStream_t stream);

}  // namespace brotensor::detail::cuda::mma_gemm
