#pragma once

// Wave 2 INT8W conv2d & auxiliary MIR builders replacing:
//   conv2d_int8w_wmma.cu (2-stage mma.sync.m16n8k16 implicit-GEMM for 3x3s1, 1x1s1, 3x3s2)
//   int8_quant.cu kernels (matmul_int8w_fp16 & general grouped/dilated conv2d_int8w_fp16)

#include <brass/mir/module.hpp>

#include <cstdint>

namespace brotensor::kernels {

// cfg_id: 0 = (3,3,1,1,1,1), 1 = (1,1,0,0,1,1), 2 = (3,3,1,1,2,2).
// Params: X(0) W_int8(1) scales(2) bias(3) Y(4)
//         | N(5) C_in(6) H(7) W(8) C_out(9) H_out(10) W_out(11) M_tot(12) HW_out(13) K_tot(14)
brass::Function* build_conv2d_int8w_mma_mir(brass::Module& mod, std::uint8_t cfg_id,
                                            bool has_bias, bool aligned_k8);

// 16x16 tiled W_int8(M, K) * scales(M) @ X(K, N) -> Y(M, N) FP16.
// Params: W(0) scales(1) X(2) Y(3) | M(4) N(5) K(6)
brass::Function* build_matmul_int8w_fp16_mir(brass::Module& mod);

// General grouped/dilated INT8W conv2d forward fallback (FP16).
// Params: X(0) W(1) scales(2) bias(3) Y(4)
//         | N(5) C_in(6) H(7) W_in(8) C_out(9) kH(10) kW(11) H_out(12) W_out(13)
//           sh(14) sw(15) ph(16) pw(17) dh(18) dw(19) Cg_in(20) Cg_out(21) total(22)
brass::Function* build_conv2d_int8w_fallback_mir(brass::Module& mod, bool has_bias);

}  // namespace brotensor::kernels
