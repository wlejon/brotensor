// ─── CUDA codec quantization ops (brosoundml CHUNK 5, family D) ─────────────
//
// CUDA port of src/cpu/codec_quant.cpp. FP32-only, mirroring the CPU contracts:
//   vq_encode_forward / vq_encode_backward        — vector quantization
//   fsq_quantize_forward / fsq_quantize_backward  — finite scalar quantization
//
// INT32 outputs (vq_encode indices, fsq packed_indices) are resized AND
// dtype-set to INT32; Tensor::data is treated as an int32_t* device pointer.
//
// Accumulation:
//   *_forward             — all outputs OVERWRITTEN.
//   *_backward            — dX OVERWRITTEN (straight-through identity passthru).

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor::detail::cuda {

namespace {

constexpr int CQ_BLOCK = 128;

inline int cq_grid(long long n) {
    long long blocks = (n + CQ_BLOCK - 1) / CQ_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void require_fp32(const char* op, const ::brotensor::Tensor& t,
                         const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32 (audio ops are FP32-only)");
    }
}

inline void require_int32(const char* op, const ::brotensor::Tensor& t,
                          const char* name) {
    if (t.dtype != ::brotensor::Dtype::INT32) {
        fail(op, std::string(name) + " must be INT32");
    }
}

// ─── vq_encode_forward ──────────────────────────────────────────────────────
// One thread per row n: nearest-codeword search (lowest index wins ties).
__global__ void vq_encode_forward_kernel(const float* __restrict__ x,
                                         const float* __restrict__ codebook,
                                         int N, int D, int K,
                                         int32_t* __restrict__ indices,
                                         float* __restrict__ quantized) {
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N;
         n += blockDim.x * gridDim.x) {
        const float* x_row = x + (long long)n * D;
        float best_d2 = 3.4028235e38f;
        int best_k = 0;
        for (int k = 0; k < K; ++k) {
            const float* c_row = codebook + (long long)k * D;
            float d2 = 0.0f;
            for (int j = 0; j < D; ++j) {
                const float diff = x_row[j] - c_row[j];
                d2 += diff * diff;
            }
            if (d2 < best_d2) { best_d2 = d2; best_k = k; }  // strict < : low idx
        }
        indices[n] = best_k;
        const float* c_best = codebook + (long long)best_k * D;
        float* q_row = quantized + (long long)n * D;
        for (int j = 0; j < D; ++j) q_row[j] = c_best[j];
    }
}

// ─── fsq_quantize_forward ───────────────────────────────────────────────────
// One thread per row n. Mixed-radix pack, dimension 0 the least-significant
// digit, built MSD-down via Horner's scheme.
__global__ void fsq_quantize_forward_kernel(const float* __restrict__ x,
                                            const int32_t* __restrict__ levels,
                                            int N, int D,
                                            float* __restrict__ quantized,
                                            int32_t* __restrict__ packed) {
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N;
         n += blockDim.x * gridDim.x) {
        const float* x_row = x + (long long)n * D;
        float* q_row = quantized + (long long)n * D;
        long long code = 0;
        for (int d = D - 1; d >= 0; --d) {
            const int L = levels[d];
            const float h = static_cast<float>(L - 1) * 0.5f;
            float v = x_row[d];
            if (v < -1.0f) v = -1.0f;
            else if (v > 1.0f) v = 1.0f;
            float idx_f = roundf((v + 1.0f) * 0.5f * static_cast<float>(L - 1));
            int idx = static_cast<int>(idx_f);
            if (idx < 0) idx = 0;
            else if (idx > L - 1) idx = L - 1;
            q_row[d] = static_cast<float>(idx) / h - 1.0f;
            code = code * static_cast<long long>(L) +
                   static_cast<long long>(idx);
        }
        packed[n] = static_cast<int32_t>(code);
    }
}

// Straight-through identity passthrough — dX = dQuantized. Alias-safe.
__global__ void ste_copy_kernel(const float* __restrict__ src,
                                float* __restrict__ dst, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════
//  Wrappers
// ════════════════════════════════════════════════════════════════════════════

void vq_encode_forward(const ::brotensor::Tensor& x,
                       const ::brotensor::Tensor& codebook,
                       ::brotensor::Tensor& indices,
                       ::brotensor::Tensor& quantized) {
    require_fp32("vq_encode_forward", x, "x");
    require_fp32("vq_encode_forward", codebook, "codebook");
    const int N = x.rows;
    const int D = x.cols;
    const int K = codebook.rows;
    if (codebook.cols != D) {
        fail("vq_encode_forward", "codebook must have the same column count as x");
    }
    if (K == 0 && N != 0) {
        fail("vq_encode_forward", "codebook must have at least one codeword");
    }
    if (indices.rows != N || indices.cols != 1 ||
        indices.dtype != ::brotensor::Dtype::INT32) {
        indices.resize(N, 1, ::brotensor::Dtype::INT32);
    }
    if (quantized.rows != N || quantized.cols != D ||
        quantized.dtype != ::brotensor::Dtype::FP32) {
        quantized.resize(N, D, ::brotensor::Dtype::FP32);
    }
    if (N == 0) return;
    vq_encode_forward_kernel<<<cq_grid(N), CQ_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(x.data),
        static_cast<const float*>(codebook.data),
        N, D, K,
        static_cast<int32_t*>(indices.data),
        static_cast<float*>(quantized.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void vq_encode_backward(const ::brotensor::Tensor& dQuantized,
                        ::brotensor::Tensor& dX) {
    require_fp32("vq_encode_backward", dQuantized, "dQuantized");
    if (dX.rows != dQuantized.rows || dX.cols != dQuantized.cols ||
        dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(dQuantized.rows, dQuantized.cols, ::brotensor::Dtype::FP32);
    }
    const long long total = dQuantized.size();
    if (total == 0) return;
    ste_copy_kernel<<<cq_grid(total), CQ_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(dQuantized.data),
        static_cast<float*>(dX.data), total);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void fsq_quantize_forward(const ::brotensor::Tensor& x,
                          const ::brotensor::Tensor& levels,
                          ::brotensor::Tensor& quantized,
                          ::brotensor::Tensor& packed_indices) {
    require_fp32("fsq_quantize_forward", x, "x");
    require_int32("fsq_quantize_forward", levels, "levels");
    const int N = x.rows;
    const int D = x.cols;
    if (levels.size() != D) {
        fail("fsq_quantize_forward",
             "levels must have D elements (one per column of x)");
    }
    if (quantized.rows != N || quantized.cols != D ||
        quantized.dtype != ::brotensor::Dtype::FP32) {
        quantized.resize(N, D, ::brotensor::Dtype::FP32);
    }
    if (packed_indices.rows != N || packed_indices.cols != 1 ||
        packed_indices.dtype != ::brotensor::Dtype::INT32) {
        packed_indices.resize(N, 1, ::brotensor::Dtype::INT32);
    }
    if (N == 0 || D == 0) return;
    // Validate level counts (every L_d >= 2) — copy the small levels vector to
    // the host, matching the CPU op's up-front check.
    std::vector<int32_t> host_levels(static_cast<size_t>(D));
    BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(
        host_levels.data(), levels.data,
        static_cast<size_t>(D) * sizeof(int32_t), cudaMemcpyDeviceToHost,
        cur_stream()));
    BROTENSOR_CUDA_CHECK(cudaStreamSynchronize(cur_stream()));
    for (int d = 0; d < D; ++d) {
        if (host_levels[d] < 2) {
            fail("fsq_quantize_forward", "every level count must be >= 2");
        }
    }
    fsq_quantize_forward_kernel<<<cq_grid(N), CQ_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(x.data),
        static_cast<const int32_t*>(levels.data),
        N, D,
        static_cast<float*>(quantized.data),
        static_cast<int32_t*>(packed_indices.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void fsq_quantize_backward(const ::brotensor::Tensor& dQuantized,
                           ::brotensor::Tensor& dX) {
    require_fp32("fsq_quantize_backward", dQuantized, "dQuantized");
    if (dX.rows != dQuantized.rows || dX.cols != dQuantized.cols ||
        dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(dQuantized.rows, dQuantized.cols, ::brotensor::Dtype::FP32);
    }
    const long long total = dQuantized.size();
    if (total == 0) return;
    ste_copy_kernel<<<cq_grid(total), CQ_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(dQuantized.data),
        static_cast<float*>(dX.data), total);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_codec_quant(::brotensor::detail::OpsVTable& v) {
    v.vq_encode_forward     = &vq_encode_forward;
    v.vq_encode_backward    = &vq_encode_backward;
    v.fsq_quantize_forward  = &fsq_quantize_forward;
    v.fsq_quantize_backward = &fsq_quantize_backward;
}

} // namespace brotensor::detail::cuda
