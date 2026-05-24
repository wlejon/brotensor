// ─── CUDA row gather + scatter-add ──────────────────────────────────────────
//
// FP32-only port of src/cpu/gather_rows.cpp. Contracts mirror CPU exactly:
//   * Idx is INT32 shape (M, 1).
//   * gather_rows      : Y[m, :] = X[Idx[m], :], Y OVERWRITTEN.
//   * scatter_rows_add : dX[Idx[m], :] += dY[m, :]; we zero dX first, then
//                        atomicAdd accumulate (duplicate indices in Idx
//                        accumulate — adjoint of gather_rows).
//
// Bounds: OOB Idx is UB on all backends — caller owns the precondition.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

namespace {

constexpr int GS_BLOCK = 256;

inline int gs_grid(long long n) {
    long long blocks = (n + GS_BLOCK - 1) / GS_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
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

inline void check_idx(const ::brotensor::Tensor& Idx, const char* op) {
    if (Idx.dtype != ::brotensor::Dtype::INT32) {
        fail(op, "Idx must be INT32");
    }
    if (Idx.cols != 1) {
        fail(op, "Idx must be shaped (M, 1)");
    }
}

// Kernel-per-output-element: one thread copies one (m, c) cell.
__global__ void gather_rows_kernel(const float* __restrict__ X,
                                   const int32_t* __restrict__ Idx,
                                   float* __restrict__ Y,
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

// Kernel-per-input-element of dY: scatter-add into dX[Idx[m], c].
__global__ void scatter_rows_add_kernel(const float* __restrict__ dY,
                                        const int32_t* __restrict__ Idx,
                                        float* __restrict__ dX,
                                        int M, int C) {
    const long long total = (long long)M * C;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int c = static_cast<int>(idx % C);
        const int m = static_cast<int>(idx / C);
        const int r = Idx[m];
        atomicAdd(&dX[(long long)r * C + c], dY[idx]);
    }
}

} // namespace

void gather_rows(const ::brotensor::Tensor& X,
                 const ::brotensor::Tensor& Idx,
                 ::brotensor::Tensor& Y) {
    const char* op = "gather_rows";
    check_fp32(X, op, "X");
    check_idx(Idx, op);
    const int M = Idx.rows;
    const int C = X.cols;
    if (Y.rows != M || Y.cols != C || Y.dtype != ::brotensor::Dtype::FP32) {
        Y.resize(M, C, ::brotensor::Dtype::FP32);
    }
    if (M == 0 || C == 0) return;

    const long long total = (long long)M * C;
    gather_rows_kernel<<<gs_grid(total), GS_BLOCK>>>(
        static_cast<const float*>(X.data),
        static_cast<const int32_t*>(Idx.data),
        static_cast<float*>(Y.data),
        M, C);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void scatter_rows_add(const ::brotensor::Tensor& dY,
                      const ::brotensor::Tensor& Idx, int R,
                      ::brotensor::Tensor& dX) {
    const char* op = "scatter_rows_add";
    check_fp32(dY, op, "dY");
    check_idx(Idx, op);
    if (R < 0) fail(op, "R must be >= 0");
    const int M = Idx.rows;
    if (dY.rows != M) {
        fail(op, "dY.rows must equal Idx.rows");
    }
    const int C = dY.cols;
    if (dX.rows != R || dX.cols != C || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(R, C, ::brotensor::Dtype::FP32);
    }
    if (R == 0 || C == 0) return;

    BROTENSOR_CUDA_CHECK(cudaMemset(
        dX.data, 0, static_cast<size_t>((long long)R * C) * sizeof(float)));

    if (M == 0) return;

    const long long total = (long long)M * C;
    scatter_rows_add_kernel<<<gs_grid(total), GS_BLOCK>>>(
        static_cast<const float*>(dY.data),
        static_cast<const int32_t*>(Idx.data),
        static_cast<float*>(dX.data),
        M, C);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_gather_scatter(::brotensor::detail::OpsVTable& v) {
    v.gather_rows      = &gather_rows;
    v.scatter_rows_add = &scatter_rows_add;
}

} // namespace brotensor::detail::cuda
