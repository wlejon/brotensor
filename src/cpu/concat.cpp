// ─── CPU concat ops (CHUNK 1) ──────────────────────────────────────────────
//
// FP32 scalar host implementations. Ports the GPU kernels in
// src/cuda/concat.cu — same layout/contracts, FP32 path only.
//
//   concat_batched_rows          — concat parts along the column axis; all
//                                  parts share row count B, out is
//                                  (B, sum cols).
//   concat_nchw_channels         — concat NCHW tensors along the channel
//                                  axis; out is (N, total_C*H*W).
//   concat_nchw_channels_backward— inverse of concat_nchw_channels; slices
//                                  dY back into per-part tensors (overwrite).

#include <brotensor/tensor.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace brotensor::detail::cpu {

void concat_batched_rows(const std::vector<const ::brotensor::Tensor*>& parts,
                         ::brotensor::Tensor& out) {
    if (parts.empty()) { out.resize(0, 0, ::brotensor::Dtype::FP32); return; }
    int B = 0;
    int total_cols = 0;
    for (const auto* p : parts) {
        if (!p) continue;
        if (B == 0) B = p->rows;
        total_cols += p->cols;
    }
    if (out.rows != B || out.cols != total_cols ||
        out.dtype != ::brotensor::Dtype::FP32) {
        out.resize(B, total_cols, ::brotensor::Dtype::FP32);
    }
    if (B == 0 || total_cols == 0) return;

    float* op = out.host_f32_mut();
    int col_off = 0;
    for (const auto* p : parts) {
        if (!p) continue;
        const int d = p->cols;
        if (d == 0) continue;
        const float* pp = p->host_f32();
        for (int b = 0; b < B; ++b) {
            const float* src = pp + static_cast<std::size_t>(b) * d;
            float* dst = op + static_cast<std::size_t>(b) * total_cols + col_off;
            for (int j = 0; j < d; ++j) dst[j] = src[j];
        }
        col_off += d;
    }
}

void concat_nchw_channels(const std::vector<const ::brotensor::Tensor*>& parts,
                          int N, int H, int W,
                          const std::vector<int>& C_per_part,
                          ::brotensor::Tensor& out) {
    if (parts.size() != C_per_part.size()) {
        throw std::runtime_error("concat_nchw_channels: parts.size() != C_per_part.size()");
    }
    int total_C = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const auto* p = parts[i];
        const int Ci = C_per_part[i];
        if (!p) throw std::runtime_error("concat_nchw_channels: null part");
        if (p->size() != static_cast<int>(
                static_cast<std::size_t>(N) * Ci * H * W)) {
            throw std::runtime_error("concat_nchw_channels: part size mismatch (expected N*C_i*H*W)");
        }
        total_C += Ci;
    }
    const int total_cols = total_C * H * W;
    if (out.rows != N || out.cols != total_cols ||
        out.dtype != ::brotensor::Dtype::FP32) {
        out.resize(N, total_cols, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || total_cols == 0) return;

    const std::size_t HW = static_cast<std::size_t>(H) * W;
    float* op = out.host_f32_mut();
    std::size_t c_off = 0;  // running channel offset
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const int Ci = C_per_part[i];
        if (Ci == 0) continue;
        const std::size_t part_chunk = static_cast<std::size_t>(Ci) * HW;
        const float* pp = parts[i]->host_f32();
        for (int n = 0; n < N; ++n) {
            const float* src = pp + static_cast<std::size_t>(n) * part_chunk;
            float* dst = op + static_cast<std::size_t>(n) * total_cols
                            + c_off * HW;
            for (std::size_t k = 0; k < part_chunk; ++k) dst[k] = src[k];
        }
        c_off += static_cast<std::size_t>(Ci);
    }
}

void concat_nchw_channels_backward(const ::brotensor::Tensor& dY,
                                   int N, int H, int W,
                                   const std::vector<int>& C_per_part,
                                   const std::vector<::brotensor::Tensor*>& parts) {
    if (parts.size() != C_per_part.size()) {
        throw std::runtime_error("concat_nchw_channels_backward: parts.size() != C_per_part.size()");
    }
    int total_C = 0;
    for (int Ci : C_per_part) total_C += Ci;
    const int expected_cols = total_C * H * W;
    if (dY.rows != N || dY.cols != expected_cols) {
        throw std::runtime_error("concat_nchw_channels_backward: dY shape mismatch (expected N x total_C*H*W)");
    }

    const std::size_t HW = static_cast<std::size_t>(H) * W;
    const float* dyp = dY.host_f32();
    std::size_t c_off = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        ::brotensor::Tensor* p = parts[i];
        const int Ci = C_per_part[i];
        if (!p) throw std::runtime_error("concat_nchw_channels_backward: null part");
        const int cols = Ci * H * W;
        if (p->rows != N || p->cols != cols ||
            p->dtype != ::brotensor::Dtype::FP32) {
            p->resize(N, cols, ::brotensor::Dtype::FP32);
        }
        if (Ci == 0 || N == 0 || HW == 0) {
            c_off += static_cast<std::size_t>(Ci);
            continue;
        }
        const std::size_t part_chunk = static_cast<std::size_t>(Ci) * HW;
        float* pp = p->host_f32_mut();
        for (int n = 0; n < N; ++n) {
            const float* src = dyp + static_cast<std::size_t>(n) * expected_cols
                                   + c_off * HW;
            float* dst = pp + static_cast<std::size_t>(n) * part_chunk;
            for (std::size_t k = 0; k < part_chunk; ++k) dst[k] = src[k];
        }
        c_off += static_cast<std::size_t>(Ci);
    }
}

} // namespace brotensor::detail::cpu
