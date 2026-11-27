#pragma once

// Parameterized 2-stage Tensor Core quantized GEMM MIR kernel builder:
//   Y(B, M) = X(B, K) @ dequant(W(M, K))^T [+ bias(M)]
// Supports QF_INT8 (FP16/BF16 activations), QF_Q8_0, QF_Q4K, and QF_Q6K (FP16).

#include <brotensor/tensor.h>

#include <brass/mir/module.hpp>

#include <cstdint>

namespace brotensor::kernels {

struct GemmQuantMmaCfg {
    std::uint32_t bm = 64;
    std::uint32_t bn = 64;
    std::uint32_t warps_m = 2;
    std::uint32_t warps_n = 2;
    std::uint32_t threads = 128;
    std::uint32_t min_ctas_per_sm = 2;
};

inline constexpr GemmQuantMmaCfg kGemmQuantMmaCfgs[2] = {
    {64u,  64u, 2u, 2u, 128u, 3u},  // cfg=0: smaller B (<= 64)
    {128u, 64u, 4u, 2u, 256u, 2u},  // cfg=1: larger B (> 64, 2x weight reuse, 256 threads)
};

struct GemmQuantMmaSpec {
    std::uint8_t fmt = 1;       // QF_INT8 / QF_Q8_0 / QF_Q4K / QF_Q6K
    Dtype dt = Dtype::FP16;     // FP16 or BF16 (BF16 for QF_INT8 only)
    std::uint8_t cfg = 0;       // 0 or 1
    bool has_bias = false;
    bool aligned_mnk = false;   // B % BM == 0 && M % 64 == 0 && K % 32 == 0
    bool swizzle_l2 = false;    // L2 panel swizzle across M/N tiles
};

// Params: X(0) W(1) scales(2) bias(3) Y(4) | B(5) M(6) K(7) row_bytes(8) grid_m(9)
brass::Function* build_gemm_quant_mma_mir(brass::Module& mod, const GemmQuantMmaSpec& spec);

}  // namespace brotensor::kernels
