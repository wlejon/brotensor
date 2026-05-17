#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#import "internal.h"

#include <cstring>

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
