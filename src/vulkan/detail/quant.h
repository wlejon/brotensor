#pragma once

// Quantised weights on Vulkan (ops_quant.cpp): INT8 with per-row FP32 scales
// (W8A16) and the GGUF block formats Q8_0, Q4_K, Q6_K. A weight is never
// expanded to FP16 on the matrix paths: the decode kernels read it raw
// (gemv_q.comp for <= 8 activation rows, gemm_cm.comp's QB tiles for more).
// Design: docs/vulkan.md "Quantised weights".

#include "device.h"

#include <brotensor/tensor.h>

#include <cstdint>

namespace brotensor::detail::vulkan {

struct QuantW {
    int fmt = 0;                  // QF_* (shaders/op_codes.h)
    std::uint64_t w = 0;          // device address of the first row
    std::uint64_t scale = 0;      // QF_INT8: (rows) FP32 scales
    int rows = 0, k = 0;          // output rows, elements per row
    std::uint32_t rowbytes = 0;   // bytes per row
};

// Validates W (and, for INT8, scales) and describes it. INT8 weights are
// (rows, k) Dtype::INT8 with (rows, 1) FP32 scales; GGUF weights carry their
// own dtype, k a multiple of the block.
QuantW quant_weight(const ::brotensor::Tensor& W, const ::brotensor::Tensor* scales, const char* op);

// W as FP16 (rows, k), row-major, written to `y`.
void dequant(DeviceCtx& d, const QuantW& q, std::uint64_t y);

// Y(B, rows) = X(B, k) W^T + bias, X / Y / bias in `dt` (FP16 or BF16),
// row pitches ldx / ldy in elements, bias 0 for none.
void quant_linear(DeviceCtx& d, const QuantW& q, std::uint64_t x, ::brotensor::Dtype dt, int B, int ldx,
                  std::uint64_t bias, std::uint64_t y, int ldy, const char* op);

// Which kernel quant_linear() uses for B rows: "gemv", "coopmat" or
// "dequant" (expand to FP16 once, then the dense GEMM: devices without
// cooperative matrix).
const char* quant_linear_path(DeviceCtx& d, int B);

}  // namespace brotensor::detail::vulkan
