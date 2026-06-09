// ─── CUDA row gather + scatter-add ──────────────────────────────────────────
//
// CUDA port of src/cpu/gather_rows.cpp. Contracts:
//   * Idx is INT32 shape (M, 1).
//   * gather_rows      : Y[m, :] = X[Idx[m], :], Y OVERWRITTEN.
//   * scatter_rows_add : dX[Idx[m], :] += dY[m, :]; we zero dX first, then
//                        atomicAdd accumulate (duplicate indices in Idx
//                        accumulate — adjoint of gather_rows).
//
// Bounds: OOB Idx is UB on all backends — caller owns the precondition.
//
// CPU is FP32-only; CUDA additionally supports FP16/BF16 on the *value*
// tensors X/Y/dY/dX. Idx remains INT32 (Rule 3 — index tensors don't suffer
// the dtype-flip bug). scatter_rows_add accumulates in FP32 scratch for
// FP16/BF16 (no portable atomicAdd on those dtypes) then casts into dX.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

namespace {

constexpr int GS_BLOCK = 256;

// Current CUDA stream — so gather/scatter inside a CUDA-graph capture region
// joins the capture stream rather than the legacy default stream (capture
// rejects the latter). Off-capture this is null = the default stream.
inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

inline int gs_grid(long long n) {
    long long blocks = (n + GS_BLOCK - 1) / GS_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void check_fp(const ::brotensor::Tensor& t,
                     const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32 &&
        t.dtype != ::brotensor::Dtype::FP16 &&
        t.dtype != ::brotensor::Dtype::BF16) {
        fail(op, std::string(name) + " must be FP32/FP16/BF16");
    }
}

inline void check_idx(const ::brotensor::Tensor& Idx, const char* op) {
    if (Idx.dtype != ::brotensor::Dtype::INT32) {
        fail(op, "Idx must be INT32");
    }
    if (Idx.cols != 1) {
        fail(op, "Idx must be shaped (M, 1)");
    }
}

template <typename T> __device__ inline float gs_load(const T* p);
template <> __device__ inline float gs_load<float>(const float* p) { return *p; }
template <> __device__ inline float gs_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float gs_load<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ inline void gs_store(T* p, float v);
template <> __device__ inline void gs_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void gs_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void gs_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// Kernel-per-output-element: one thread copies one (m, c) cell.
template <typename T>
__global__ void gather_rows_kernel(const T* __restrict__ X,
                                   const int32_t* __restrict__ Idx,
                                   T* __restrict__ Y,
                                   int M, int C) {
    const long long total = (long long)M * C;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int c = static_cast<int>(idx % C);
        const int m = static_cast<int>(idx / C);
        const int r = Idx[m];
        Y[idx] = X[(long long)r * C + c];
    }
}

// scatter into FP32 destination (used directly for FP32; via scratch for fp16/bf16)
template <typename T>
__global__ void scatter_rows_add_into_fp32_kernel(const T* __restrict__ dY,
                                                  const int32_t* __restrict__ Idx,
                                                  float* __restrict__ dX_fp32,
                                                  int M, int C) {
    const long long total = (long long)M * C;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int c = static_cast<int>(idx % C);
        const int m = static_cast<int>(idx / C);
        const int r = Idx[m];
        atomicAdd(&dX_fp32[(long long)r * C + c], gs_load<T>(&dY[idx]));
    }
}

template <typename T>
__global__ void gs_cast_fp32_to_T(const float* __restrict__ src,
                                  T* __restrict__ dst, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        gs_store<T>(&dst[i], src[i]);
    }
}

inline size_t bytes_of(::brotensor::Dtype d) {
    return ::brotensor::dtype_size_bytes(d);
}

} // namespace

void gather_rows(const ::brotensor::Tensor& X,
                 const ::brotensor::Tensor& Idx,
                 ::brotensor::Tensor& Y) {
    const char* op = "gather_rows";
    check_fp(X, op, "X");
    check_idx(Idx, op);
    const int M = Idx.rows;
    const int C = X.cols;
    if (Y.rows != M || Y.cols != C || Y.dtype != X.dtype) {
        Y.resize(M, C, X.dtype);
    }
    if (M == 0 || C == 0) return;

    const long long total = (long long)M * C;
    if (X.dtype == ::brotensor::Dtype::FP16) {
        gather_rows_kernel<__half><<<gs_grid(total), GS_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data),
            static_cast<const int32_t*>(Idx.data),
            static_cast<__half*>(Y.data), M, C);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        gather_rows_kernel<__nv_bfloat16><<<gs_grid(total), GS_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<const int32_t*>(Idx.data),
            static_cast<__nv_bfloat16*>(Y.data), M, C);
    } else {
        gather_rows_kernel<float><<<gs_grid(total), GS_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X.data),
            static_cast<const int32_t*>(Idx.data),
            static_cast<float*>(Y.data), M, C);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void scatter_rows_add(const ::brotensor::Tensor& dY,
                      const ::brotensor::Tensor& Idx, int R,
                      ::brotensor::Tensor& dX) {
    const char* op = "scatter_rows_add";
    check_fp(dY, op, "dY");
    check_idx(Idx, op);
    if (R < 0) fail(op, "R must be >= 0");
    const int M = Idx.rows;
    if (dY.rows != M) {
        fail(op, "dY.rows must equal Idx.rows");
    }
    const int C = dY.cols;
    if (dX.rows != R || dX.cols != C || dX.dtype != dY.dtype) {
        dX.resize(R, C, dY.dtype);
    }
    if (R == 0 || C == 0) return;

    const long long total_dst = (long long)R * C;
    if (dY.dtype == ::brotensor::Dtype::FP32) {
        BROTENSOR_CUDA_CHECK(cudaMemset(
            dX.data, 0, static_cast<size_t>(total_dst) * sizeof(float)));
        if (M == 0) return;
        const long long total = (long long)M * C;
        scatter_rows_add_into_fp32_kernel<float><<<gs_grid(total), GS_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(dY.data),
            static_cast<const int32_t*>(Idx.data),
            static_cast<float*>(dX.data), M, C);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }

    // FP16/BF16: accumulate into FP32 scratch, then cast (overwrite) dX.
    float* d_scratch = nullptr;
    BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scratch),
                                    static_cast<size_t>(total_dst) * sizeof(float)));
    BROTENSOR_CUDA_CHECK(cudaMemset(d_scratch, 0,
                                    static_cast<size_t>(total_dst) * sizeof(float)));
    if (M > 0) {
        const long long total = (long long)M * C;
        if (dY.dtype == ::brotensor::Dtype::FP16) {
            scatter_rows_add_into_fp32_kernel<__half><<<gs_grid(total), GS_BLOCK, 0, cur_stream()>>>(
                static_cast<const __half*>(dY.data),
                static_cast<const int32_t*>(Idx.data),
                d_scratch, M, C);
        } else {
            scatter_rows_add_into_fp32_kernel<__nv_bfloat16><<<gs_grid(total), GS_BLOCK, 0, cur_stream()>>>(
                static_cast<const __nv_bfloat16*>(dY.data),
                static_cast<const int32_t*>(Idx.data),
                d_scratch, M, C);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    if (dY.dtype == ::brotensor::Dtype::FP16) {
        gs_cast_fp32_to_T<__half><<<gs_grid(total_dst), GS_BLOCK, 0, cur_stream()>>>(
            d_scratch, static_cast<__half*>(dX.data), total_dst);
    } else {
        gs_cast_fp32_to_T<__nv_bfloat16><<<gs_grid(total_dst), GS_BLOCK, 0, cur_stream()>>>(
            d_scratch, static_cast<__nv_bfloat16*>(dX.data), total_dst);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    cudaFree(d_scratch);
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_gather_scatter(::brotensor::detail::OpsVTable& v) {
    v.gather_rows      = &gather_rows;
    v.scatter_rows_add = &scatter_rows_add;
}

} // namespace brotensor::detail::cuda
