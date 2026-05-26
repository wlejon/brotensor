// ─── CPU embedding lookup ops (CHUNK 1) ────────────────────────────────────
//
// FP32 scalar host implementations. Ports the GPU kernels in
// src/cuda/embedding.cu — FP32 path only.
//
// `d_idx` is a host int32 pointer on the CPU backend (the dispatcher keeps
// raw-pointer args on the host).
//
//   embedding_lookup_forward  — gather B rows from `table` by index.
//   embedding_lookup_backward — scatter-accumulate dOut rows into dTable.
//                               ACCUMULATES (+=); caller zeros dTable.

#include <brotensor/tensor.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace brotensor::detail::cpu {

void embedding_lookup_forward(const ::brotensor::Tensor& table,
                              const int32_t* d_idx, int B,
                              ::brotensor::Tensor& out) {
    if (table.dtype != ::brotensor::Dtype::FP32) {
        throw std::runtime_error(
            "brotensor: embedding_lookup_forward: CPU backend requires "
            "FP32 table (quantized tables must be dequantized first)");
    }
    const int D = table.cols;
    if (out.rows != B || out.cols != D ||
        out.dtype != ::brotensor::Dtype::FP32) {
        out.resize(B, D, ::brotensor::Dtype::FP32);
    }
    const int total = B * D;
    if (total == 0) return;

    const float* tp = table.host_f32();
    float* op = out.host_f32_mut();
    for (int b = 0; b < B; ++b) {
        const int row = d_idx[b];
        const float* src = tp + static_cast<std::size_t>(row) * D;
        float* dst = op + static_cast<std::size_t>(b) * D;
        for (int j = 0; j < D; ++j) dst[j] = src[j];
    }
}

void embedding_lookup_backward(const ::brotensor::Tensor& dOut,
                               const int32_t* d_idx, int B,
                               ::brotensor::Tensor& dTable) {
    const int D = dTable.cols;
    const int total = B * D;
    if (total == 0) return;

    const float* dop = dOut.host_f32();
    float* dtp = dTable.host_f32_mut();
    // Scatter-accumulate: caller zeros dTable beforehand. Repeated indices
    // accumulate, matching the GPU atomicAdd.
    for (int b = 0; b < B; ++b) {
        const int row = d_idx[b];
        const float* src = dop + static_cast<std::size_t>(b) * D;
        float* dst = dtp + static_cast<std::size_t>(row) * D;
        for (int j = 0; j < D; ++j) dst[j] += src[j];
    }
}

} // namespace brotensor::detail::cpu
