#pragma once

// Shared host-side validation, output geometry, polyphase filter sizing, and
// threadgroup/shared-memory eligibility for filtered_lrelu_forward.

#include <brotensor/tensor.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace brotensor::gpu_common {

inline constexpr int kFlTileH = 16;
inline constexpr int kFlTileW = 16;
inline constexpr std::uint32_t kFlBlockThreads = static_cast<std::uint32_t>(kFlTileH * kFlTileW);

inline bool fl_is_fp(Dtype d) {
    return d == Dtype::FP32 || d == Dtype::FP16 || d == Dtype::BF16;
}

inline std::size_t fl_elem_size(Dtype d) {
    return (d == Dtype::FP32) ? 4u : 2u;
}

struct FilteredLreluPlan {
    int fuH = 0, fuW = 0, fdH = 0, fdW = 0;
    int Hu = 0, Wu = 0, Huo = 0, Wuo = 0, Hout = 0, Wout = 0;
    int JyMax = 0, JxMax = 0;
    int AH = 0, AW = 0;
    bool supported = false;
};

/// Computes filtered_lrelu output geometry and checks whether the fused kernel
/// can run natively on the calling GPU backend under `max_shmem_bytes`.
inline FilteredLreluPlan plan_filtered_lrelu(
    const Tensor& X, const Tensor& fu, const Tensor& fd, const Tensor* b,
    int N, int C, int H, int W, int up, int down,
    int pad_x0, int pad_x1, int pad_y0, int pad_y1,
    const Tensor& up_buf, const Tensor& act_buf,
    std::size_t max_shmem_bytes, bool shmem_uses_float_stage) {
    FilteredLreluPlan p{};
    p.fuH = fu.rows;
    p.fuW = fu.cols;
    p.fdH = fd.rows;
    p.fdW = fd.cols;
    p.Hu = H * up;
    p.Wu = W * up;
    p.Huo = p.Hu + pad_y0 + pad_y1 - p.fuH + 1;
    p.Wuo = p.Wu + pad_x0 + pad_x1 - p.fuW + 1;
    p.Hout = (down >= 1) ? ((p.Huo - p.fdH) / down + 1) : 0;
    p.Wout = (down >= 1) ? ((p.Wuo - p.fdW) / down + 1) : 0;

    p.JyMax = (up >= 1) ? ((p.fuH + up - 1) / up) : 0;
    p.JxMax = (up >= 1) ? ((p.fuW + up - 1) / up) : 0;
    p.AH = (kFlTileH - 1) * down + p.fdH;
    p.AW = (kFlTileW - 1) * down + p.fdW;
    const std::size_t stage_elem = shmem_uses_float_stage ? sizeof(float) : fl_elem_size(X.dtype);
    const std::size_t shmem_bytes = static_cast<std::size_t>(p.AH) * static_cast<std::size_t>(p.AW) * stage_elem;

    const bool want_cache = (up_buf.data != nullptr) || (act_buf.data != nullptr);
    const bool fits_aux =
        up >= 1 && up < 256 && down >= 1 && down < 256 &&
        p.JyMax >= 1 && p.JyMax < 256 && p.JxMax >= 1 && p.JxMax < 256 &&
        p.fdH >= 1 && p.fdH < 256 && p.fdW >= 1 && p.fdW < 256;
    p.supported =
        !want_cache && fl_is_fp(X.dtype) && fu.dtype == X.dtype && fd.dtype == X.dtype &&
        (!b || b->dtype == X.dtype) &&
        p.Huo > 0 && p.Wuo > 0 && p.Hout > 0 && p.Wout > 0 &&
        fits_aux && shmem_bytes <= max_shmem_bytes &&
        static_cast<long long>(N) * C <= 65535;
    return p;
}

inline bool prepare_filtered_lrelu_out(const Tensor& X, int N, int C, int H, int W,
                                       const FilteredLreluPlan& p, Tensor& Y) {
    if (X.rows != N || X.cols != C * H * W) {
        throw std::runtime_error("brotensor: filtered_lrelu_forward: X shape mismatch");
    }
    const int out_cols = C * p.Hout * p.Wout;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    return N > 0 && out_cols > 0;
}

inline std::uint64_t filtered_lrelu_main_aux(const FilteredLreluPlan& p, int up, int down) {
    return (static_cast<std::uint64_t>(up) & 0xffu) |
           ((static_cast<std::uint64_t>(down) & 0xffu) << 8) |
           ((static_cast<std::uint64_t>(p.JyMax) & 0xffu) << 16) |
           ((static_cast<std::uint64_t>(p.JxMax) & 0xffu) << 24) |
           ((static_cast<std::uint64_t>(p.fdH) & 0xffu) << 32) |
           ((static_cast<std::uint64_t>(p.fdW) & 0xffu) << 40);
}

}  // namespace brotensor::gpu_common
