#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#import "internal.h"

#include <cstring>
#include <stdexcept>

namespace brotensor {

using metal_impl::buffer_for;

namespace {

inline void copy_bytes(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    std::memcpy(dst, src, bytes);
}

} // namespace

void concat_rows_gpu(const std::vector<const GpuTensor*>& parts,
                     GpuTensor& out) {
    int total = 0;
    Dtype dt = Dtype::FP32;
    bool seen = false;
    for (const auto* p : parts) {
        if (!p) continue;
        total += p->size();
        if (!seen) { dt = p->dtype; seen = true; }
    }
    if (out.rows != total || out.cols != 1 || out.dtype != dt) {
        out.resize(total, 1, dt);
    }
    if (total == 0) return;

    const std::size_t elem = static_cast<std::size_t>(dtype_size_bytes(dt));
    char* dst_base = reinterpret_cast<char*>(out.data);
    std::size_t off_bytes = 0;
    for (const auto* p : parts) {
        if (!p) continue;
        const int n = p->size();
        if (n == 0) continue;
        copy_bytes(dst_base + off_bytes, p->data, elem * static_cast<std::size_t>(n));
        off_bytes += elem * static_cast<std::size_t>(n);
    }
}

void split_rows_gpu(const GpuTensor& in,
                    const std::vector<GpuTensor*>& parts) {
    const std::size_t elem = static_cast<std::size_t>(dtype_size_bytes(in.dtype));
    const char* src_base = reinterpret_cast<const char*>(in.data);
    std::size_t off_bytes = 0;
    for (auto* p : parts) {
        if (!p) continue;
        const int n = p->size();
        if (n == 0) continue;
        copy_bytes(p->data, src_base + off_bytes, elem * static_cast<std::size_t>(n));
        off_bytes += elem * static_cast<std::size_t>(n);
    }
}

void concat_batched_rows_gpu(const std::vector<const GpuTensor*>& parts,
                             GpuTensor& out) {
    if (parts.empty()) { out.resize(0, 0); return; }
    int B = 0;
    int total_cols = 0;
    Dtype dt = Dtype::FP32;
    bool seen = false;
    for (const auto* p : parts) {
        if (!p) continue;
        if (B == 0) B = p->rows;
        total_cols += p->cols;
        if (!seen) { dt = p->dtype; seen = true; }
    }
    if (out.rows != B || out.cols != total_cols || out.dtype != dt) {
        out.resize(B, total_cols, dt);
    }
    if (B == 0 || total_cols == 0) return;

    const std::size_t elem = static_cast<std::size_t>(dtype_size_bytes(dt));
    const std::size_t dst_pitch = elem * static_cast<std::size_t>(total_cols);
    char* dst_base = reinterpret_cast<char*>(out.data);
    std::size_t col_off_bytes = 0;
    for (const auto* p : parts) {
        if (!p) continue;
        const int d = p->cols;
        if (d == 0) continue;
        const std::size_t row_bytes = elem * static_cast<std::size_t>(d);
        const char* src = reinterpret_cast<const char*>(p->data);
        for (int b = 0; b < B; ++b) {
            copy_bytes(dst_base + static_cast<std::size_t>(b) * dst_pitch + col_off_bytes,
                       src + static_cast<std::size_t>(b) * row_bytes,
                       row_bytes);
        }
        col_off_bytes += row_bytes;
    }
}

void concat_nchw_channels_gpu(const std::vector<const GpuTensor*>& parts,
                              int N, int H, int W,
                              const std::vector<int>& C_per_part,
                              GpuTensor& out) {
    if (parts.size() != C_per_part.size()) {
        throw std::runtime_error("concat_nchw_channels_gpu: parts.size() != C_per_part.size()");
    }
    int total_C = 0;
    Dtype dt = Dtype::FP32;
    bool seen = false;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const auto* p = parts[i];
        const int Ci = C_per_part[i];
        if (!p) throw std::runtime_error("concat_nchw_channels_gpu: null part");
        if (!seen) { dt = p->dtype; seen = true; }
        else if (p->dtype != dt) {
            throw std::runtime_error("concat_nchw_channels_gpu: dtype mismatch across parts");
        }
        if (p->size() != static_cast<int>(static_cast<std::size_t>(N) * Ci * H * W)) {
            throw std::runtime_error("concat_nchw_channels_gpu: part size mismatch (expected N*C_i*H*W)");
        }
        total_C += Ci;
    }
    const int total_cols = total_C * H * W;
    if (out.rows != N || out.cols != total_cols || out.dtype != dt) {
        out.resize(N, total_cols, dt);
    }
    if (N == 0 || total_cols == 0) return;

    const std::size_t elem = static_cast<std::size_t>(dtype_size_bytes(dt));
    const std::size_t HW = static_cast<std::size_t>(H) * static_cast<std::size_t>(W);
    const std::size_t dst_pitch = elem * static_cast<std::size_t>(total_C) * HW;
    char* dst_base = reinterpret_cast<char*>(out.data);
    std::size_t c_off = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const int Ci = C_per_part[i];
        if (Ci == 0) continue;
        const std::size_t width_bytes = elem * static_cast<std::size_t>(Ci) * HW;
        const char* src = reinterpret_cast<const char*>(parts[i]->data);
        for (int n = 0; n < N; ++n) {
            copy_bytes(dst_base + static_cast<std::size_t>(n) * dst_pitch + c_off * HW * elem,
                       src + static_cast<std::size_t>(n) * width_bytes,
                       width_bytes);
        }
        c_off += static_cast<std::size_t>(Ci);
    }
}

void copy_d2d_gpu(const GpuTensor& src, int src_off,
                  GpuTensor& dst,       int dst_off,
                  int n) {
    if (n <= 0) return;
    const std::size_t elem = static_cast<std::size_t>(dtype_size_bytes(src.dtype));
    const char* src_base = reinterpret_cast<const char*>(src.data);
    char*       dst_base = reinterpret_cast<char*>(dst.data);
    copy_bytes(dst_base + static_cast<std::size_t>(dst_off) * elem,
               src_base + static_cast<std::size_t>(src_off) * elem,
               elem * static_cast<std::size_t>(n));
}

} // namespace brotensor
