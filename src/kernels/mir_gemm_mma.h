#pragma once

// Ampere/Ada Tensor Core GEMM MIR builder:
//   C(M, N) = epilogue(A(M, K) @ W(N, K)^T + bias)
// using cp.async.cg 3-stage ring buffer, XOR-swizzled shared tiles,
// ldmatrix.x4, mma.sync.aligned.m16n8k16.f32.{f16,bf16}, optional 2D L2 slab
// swizzling, and compile-time aligned/unaligned specialization.

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

struct GemmMmaCfg {
    std::uint32_t bm;
    std::uint32_t bn;
    std::uint32_t warps_m;
    std::uint32_t warps_n;
    std::uint32_t threads;
    std::uint32_t min_ctas_per_sm;
};

constexpr GemmMmaCfg kGemmMmaCfgs[4] = {
    {128u, 128u, 2u, 4u, 256u, 2u},  // cfg 0: 48 KB smem, 64x32 warp (4x4 m16n8)
    { 64u, 128u, 1u, 4u, 128u, 2u},  // cfg 1: 36 KB smem, 64x32 warp (4x4 m16n8)
    { 64u,  64u, 2u, 2u, 128u, 4u},  // cfg 2: 24 KB smem, 32x32 warp (2x4 m16n8)
    { 32u,  64u, 2u, 2u, 128u, 4u},  // cfg 3: 18 KB smem, 16x32 warp (1x4 m16n8)
};

struct GemmMmaSpec {
    Dtype dt = Dtype::FP16;
    std::uint8_t cfg = 0;       // 0..3
    std::uint8_t epi = 0;       // 0=store, 1=accumulate, 2=geglu
    std::uint8_t act = 0;       // 0..5 (LinearActivation)
    bool has_bias = false;
    bool has_partial = false;
    bool aligned_mnk = false;
    bool swizzle_l2 = false;
    bool fast_accum = false;    // FP16 m16n8k16 per-step f16->f32 fold
    bool batched = false;       // grid.z batching with strideA/B/C (!has_partial)
};

// Params:
//   ptrs (5): A(0), W(1), C(2), bias(3), partial(4)
//   ints (5 or 8 when batched): M(5), N(6), K(7), kps(8), grid_m(9) [, strideA(10), strideB(11), strideC(12)]
brass::Function* build_gemm_mma_mir(brass::Module& mod, const GemmMmaSpec& spec);

}  // namespace brotensor::kernels
