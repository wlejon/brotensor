#pragma once

// Direct convolution MIR builder replacing src/vulkan/shaders/conv_direct.comp:
//   mode 0 = CONVD_FWD   (conv2d / conv3d, NC(T)HW, OI(T)HW grouped weights)
//   mode 1 = CONVD_TRANS (conv_transpose2d / conv2d_backward_input gather)
// Params: (ptr x, ptr w, ptr y, ptr bias,
//          i32 n, cin, t, h, wd, cout, to, ho, wo, kt, kh, kw,
//          i32 st, sh, sw, pt, ph, pw, dt, dh, dw, groups)

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

brass::Function* build_conv_direct_mir(brass::Module& mod, std::uint32_t mode, Dtype dt, bool has_bias);

}  // namespace brotensor::kernels
