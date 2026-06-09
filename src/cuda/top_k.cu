// ─── CUDA per-row top-k ─────────────────────────────────────────────────────
//
// FP32-only port of src/cpu/top_k.cpp. Contract mirrors CPU exactly:
//   * For each row of X(R, C), select the k largest values and emit them in
//     descending order in Vals(R, k); Idx(R, k) is INT32 with the original
//     column indices.
//   * Ties are broken by smaller column index (deterministic, stable).
//   * Both Vals and Idx are OVERWRITTEN. Not differentiable; no backward.
//
// Strategy: one block per row, single-threaded selection — matches CPU
// algorithm verbatim (streaming-replacement of a working set, then insertion
// sort into descending/ascending-index order). k is small in practice, R
// drives the parallelism. The simple per-row kernel beats a fancy parallel
// top-k for the (small k, modest R) regime brodiffusion / brogameagent use.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

namespace {

inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32");
    }
}

// "(a, idx_a) precedes (b, idx_b)" in descending-value / ascending-index
// ordering — same semantics as CPU's `prefers`.
__device__ inline bool prefers(float a, int idx_a, float b, int idx_b) {
    if (a != b) return a > b;
    return idx_a < idx_b;
}

// One block per row, one thread per block (k is small; matches CPU exactly
// for tie-break determinism). Working set lives in shared memory.
__global__ void top_k_rows_kernel(const float* __restrict__ X,
                                  float* __restrict__ Vals,
                                  int32_t* __restrict__ Idx,
                                  int R, int C, int k) {
    const int r = blockIdx.x;
    if (r >= R) return;
    if (threadIdx.x != 0) return;

    extern __shared__ unsigned char smem[];
    float*   ws_v = reinterpret_cast<float*>(smem);
    int32_t* ws_i = reinterpret_cast<int32_t*>(ws_v + k);

    const float* row = X + (long long)r * C;
    float*   out_v = Vals + (long long)r * k;
    int32_t* out_i = Idx  + (long long)r * k;

    // Step 1: seed working set with first k elements (verbatim).
    for (int j = 0; j < k; ++j) {
        ws_v[j] = row[j];
        ws_i[j] = j;
    }
    int weakest = 0;
    for (int j = 1; j < k; ++j) {
        if (prefers(ws_v[weakest], ws_i[weakest], ws_v[j], ws_i[j])) {
            weakest = j;
        }
    }

    // Step 2: scan remainder; replace weakest if beaten.
    for (int c = k; c < C; ++c) {
        const float v = row[c];
        if (prefers(v, c, ws_v[weakest], ws_i[weakest])) {
            ws_v[weakest] = v;
            ws_i[weakest] = c;
            weakest = 0;
            for (int j = 1; j < k; ++j) {
                if (prefers(ws_v[weakest], ws_i[weakest], ws_v[j], ws_i[j])) {
                    weakest = j;
                }
            }
        }
    }

    // Step 3: insertion-sort survivors into descending-value / ascending-
    // index order, writing directly to output.
    for (int j = 0; j < k; ++j) {
        out_v[j] = ws_v[j];
        out_i[j] = ws_i[j];
    }
    for (int i = 1; i < k; ++i) {
        const float v = out_v[i];
        const int32_t idx = out_i[i];
        int j = i;
        while (j > 0 && prefers(v, idx, out_v[j - 1], out_i[j - 1])) {
            out_v[j] = out_v[j - 1];
            out_i[j] = out_i[j - 1];
            --j;
        }
        out_v[j] = v;
        out_i[j] = idx;
    }
}

} // namespace

void top_k_rows(const ::brotensor::Tensor& X, int k,
                ::brotensor::Tensor& Vals, ::brotensor::Tensor& Idx) {
    const char* op = "top_k_rows";
    check_fp32(X, op, "X");
    const int R = X.rows, C = X.cols;
    if (k < 1) fail(op, "k must be >= 1");
    if (k > C) fail(op, "k must be <= C (per-row length)");

    if (Vals.rows != R || Vals.cols != k ||
        Vals.dtype != ::brotensor::Dtype::FP32) {
        Vals.resize(R, k, ::brotensor::Dtype::FP32);
    }
    if (Idx.rows != R || Idx.cols != k ||
        Idx.dtype != ::brotensor::Dtype::INT32) {
        Idx.resize(R, k, ::brotensor::Dtype::INT32);
    }
    if (R == 0 || k == 0) return;

    const size_t smem_bytes =
        static_cast<size_t>(k) * (sizeof(float) + sizeof(int32_t));
    top_k_rows_kernel<<<R, 1, smem_bytes, cur_stream()>>>(
        static_cast<const float*>(X.data),
        static_cast<float*>(Vals.data),
        static_cast<int32_t*>(Idx.data),
        R, C, k);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_top_k(::brotensor::detail::OpsVTable& v) {
    v.top_k_rows = &top_k_rows;
}

} // namespace brotensor::detail::cuda
