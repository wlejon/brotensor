#pragma once

// Wave 3 auxiliary MIR builders replacing small CUDA kernels in conv2d.cu,
// conv3d.cu, conv2d_igemm_f32.cu (col2im), and resblock.cu:
//   op 0: conv2d_bwd_weight_block   (accumulates += into dWt)
//   op 1: conv2d_bwd_weight_thread  (accumulates += into dWt)
//   op 2: conv2d_bwd_bias           (accumulates += into dB)
//   op 3: conv3d_int8w_fp16
//   op 4: col2im_f32
//   op 5: resblock_add_nc_shift
//   op 6: resblock_sum_hw_per_nc

#include <brotensor/tensor.h>

#include <brass/mir/module.hpp>

#include <cstdint>

namespace brotensor::kernels {

brass::Function* build_conv_aux_mir(brass::Module& mod, std::uint32_t op, Dtype dt, std::uint8_t flags);

}  // namespace brotensor::kernels
