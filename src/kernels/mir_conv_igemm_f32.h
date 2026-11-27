#pragma once

// FP32 implicit-GEMM 2D convolution MIR builder (groups == 1, arbitrary geometry).
// Params: X(0) Wt(1) bias(2) Y(3)
//         | Cin(4) H(5) W(6) Cout(7) kH(8) kW(9) sh(10) sw(11) ph(12) pw(13)
//           dh(14) dw(15) Ho(16) Wo(17) M(18) K(19) KHW(20) HWo(21)

#include <brass/mir/module.hpp>

namespace brotensor::kernels {

brass::Function* build_conv2d_igemm_f32_mir(brass::Module& mod, bool has_bias, bool vec_store);

}  // namespace brotensor::kernels
