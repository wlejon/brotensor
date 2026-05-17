#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>

namespace brotensor {

// Concatenate flat tensors end-to-end. We use cudaMemcpyAsync per part on
// the default stream — this is a single-pass copy with no kernel launches,
// trivially correct and matches what an "optimised" kernel would do at the
// PCIe-bandwidth boundary anyway.
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

    const size_t elem = static_cast<size_t>(dtype_size_bytes(dt));
    char* dst_base = reinterpret_cast<char*>(out.data);
    size_t off_bytes = 0;
    for (const auto* p : parts) {
        if (!p) continue;
        const int n = p->size();
        if (n == 0) continue;
        BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(dst_base + off_bytes, p->data,
                                       elem * n,
                                       cudaMemcpyDeviceToDevice));
        off_bytes += elem * n;
    }
}

void split_rows_gpu(const GpuTensor& in,
                    const std::vector<GpuTensor*>& parts) {
    const size_t elem = static_cast<size_t>(dtype_size_bytes(in.dtype));
    const char* src_base = reinterpret_cast<const char*>(in.data);
    size_t off_bytes = 0;
    for (auto* p : parts) {
        if (!p) continue;
        const int n = p->size();
        if (n == 0) continue;
        BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(p->data, src_base + off_bytes,
                                       elem * static_cast<size_t>(n),
                                       cudaMemcpyDeviceToDevice));
        off_bytes += elem * static_cast<size_t>(n);
    }
}

// Batched column-block concat. Each part is (B, d_i) for the same B; out
// becomes (B, sum_i d_i) with parts laid out as column blocks per row.
// out[b, off_i + j] = parts[i][b, j]. Implemented via cudaMemcpy2DAsync per
// part — pure bandwidth, no kernel launches.
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

    const size_t elem = static_cast<size_t>(dtype_size_bytes(dt));
    const size_t dst_pitch = elem * total_cols;
    char* dst_base = reinterpret_cast<char*>(out.data);
    size_t col_off_bytes = 0;
    for (const auto* p : parts) {
        if (!p) continue;
        const int d = p->cols;
        if (d == 0) continue;
        BROTENSOR_CUDA_CHECK(cudaMemcpy2DAsync(
            dst_base + col_off_bytes, dst_pitch,
            p->data,                  elem * d,
            elem * d,                 B,
            cudaMemcpyDeviceToDevice));
        col_off_bytes += elem * d;
    }
}

// Single-stream device-to-device chunk copy. Copies `n` floats from
// src.data + src_off into dst.data + dst_off. No bounds checking.
void copy_d2d_gpu(const GpuTensor& src, int src_off,
                  GpuTensor& dst,       int dst_off,
                  int n) {
    if (n <= 0) return;
    const size_t elem = static_cast<size_t>(dtype_size_bytes(src.dtype));
    const char* src_base = reinterpret_cast<const char*>(src.data);
    char*       dst_base = reinterpret_cast<char*>(dst.data);
    BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(
        dst_base + static_cast<size_t>(dst_off) * elem,
        src_base + static_cast<size_t>(src_off) * elem,
        elem * static_cast<size_t>(n),
        cudaMemcpyDeviceToDevice));
}

} // namespace brotensor
