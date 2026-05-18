// Metal stub. The host quantiser is portable, but the W8A16 device ops are
// CUDA-only pending Metal implementation.
#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace brotensor {

void quantize_int8_per_row_host(const uint16_t* W_fp16,
                                int out, int in,
                                int8_t* W_int8_out,
                                float* scales_out) {
    if (out <= 0 || in <= 0) {
        for (int r = 0; r < out; ++r) scales_out[r] = 0.0f;
        return;
    }
    for (int r = 0; r < out; ++r) {
        const uint16_t* row = W_fp16 + static_cast<size_t>(r) * static_cast<size_t>(in);
        float amax = 0.0f;
        for (int c = 0; c < in; ++c) {
            const float v = fp16_bits_to_fp32(row[c]);
            const float a = std::fabs(v);
            if (a > amax) amax = a;
        }
        const float scale = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
        const float inv   = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        scales_out[r] = scale;
        int8_t* dst = W_int8_out + static_cast<size_t>(r) * static_cast<size_t>(in);
        for (int c = 0; c < in; ++c) {
            const float v = fp16_bits_to_fp32(row[c]);
            int q = static_cast<int>(std::lrint(v * inv));
            if (q < -127) q = -127;
            if (q >  127) q =  127;
            dst[c] = static_cast<int8_t>(q);
        }
    }
}

void matmul_int8w_fp16_gpu(const GpuTensor&, const GpuTensor&,
                           const GpuTensor&, GpuTensor&) {
    throw std::runtime_error("matmul_int8w_fp16_gpu: Metal backend not yet implemented");
}

void conv2d_int8w_fp16_forward_gpu(const GpuTensor&, const GpuTensor&,
                                   const GpuTensor&, const GpuTensor*,
                                   int, int, int, int,
                                   int, int, int,
                                   int, int,
                                   int, int,
                                   int, int, int,
                                   GpuTensor&) {
    throw std::runtime_error("conv2d_int8w_fp16_forward_gpu: Metal backend not yet implemented");
}

} // namespace brotensor
