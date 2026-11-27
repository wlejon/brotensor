#pragma once

// 2-stage Tensor Core implicit-GEMM Conv2D MIR kernel builder (FP16 / BF16):
//   Y[n, oc, oh, ow] = sum_{ic, kh, kw} X[n, ic, in_h, in_w] * Wt[oc, ic, kh, kw] [+ bias[oc]]
// Covers all 7 SD1.5 + annotator geometries (shape_id 0..6).

#include <brotensor/tensor.h>

#include <brass/mir/module.hpp>

#include <cstdint>

namespace brotensor::kernels {

struct Conv2dMmaCfg {
    std::uint32_t bm = 64;
    std::uint32_t bn = 64;
    std::uint32_t warps_m = 2;
    std::uint32_t warps_n = 2;
    std::uint32_t threads = 128;
    std::uint32_t min_ctas_per_sm = 3;
};

inline constexpr Conv2dMmaCfg kConv2dMmaCfgs[2] = {
    {64u,  64u, 2u, 2u, 128u, 3u},  // cfg=0: 4 warps, 16 KB smem
    {128u, 64u, 4u, 2u, 256u, 2u},  // cfg=1: 8 warps, 24 KB smem (larger M)
};

struct Conv2dMmaSpec {
    Dtype dt = Dtype::FP16;         // FP16 or BF16
    std::uint8_t shape_id = 0;      // 0..6: 3x3p1s1, 1x1p0s1, 3x3p1s2, 3x3p0s1, 7x7p3s1, 7x7p0s1, 5x5p2s1
    std::uint8_t cfg = 0;           // 0 or 1
    bool has_bias = false;
    bool aligned_k8 = false;        // (K_tot & 7) == 0 -> cp.async.cg.16 on Wt
};

// Params: X(0) Wt(1) bias(2) Y(3)
//         | N(4) C_in(5) H(6) W(7) C_out(8) H_out(9) W_out(10) M_tot(11) HW_out(12) K_tot(13)
brass::Function* build_conv2d_mma_mir(brass::Module& mod, const Conv2dMmaSpec& spec);

}  // namespace brotensor::kernels
