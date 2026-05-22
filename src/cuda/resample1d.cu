// ─── CUDA 1D resampling ops (CHUNK 6, family E) ─────────────────────────────
//
// CUDA port of src/cpu/resample1d.cpp. FP32-only. Arbitrary-scale resampling
// along the length axis of an NCL audio tensor.
//
// Memory layout (NCL flat): X / Y / dX / dY indexed ((n*C + c)*L) + l.
//
// Sampling convention — PyTorch align_corners=False:
//   src = (dst + 0.5) * (L_in / L_out) - 0.5
//   nearest : Y[dst] = X[ clamp(round_half_to_even(src), 0, L_in-1) ]
//   linear  : s = clamp(src, 0, L_in-1), x0 = floor(s), x1 = min(x0+1, L_in-1),
//             f = s - x0,  Y[dst] = (1-f)*X[x0] + f*X[x1]
//
// Accumulation:
//   resample1d_forward  — Y  OVERWRITTEN.
//   resample1d_backward — dX OVERWRITTEN (zeroed, then scatter-add via the
//                         same weights — atomicAdd handles tap collisions).

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

namespace {

constexpr int RS_BLOCK = 256;

inline int rs_grid(long long n) {
    long long blocks = (n + RS_BLOCK - 1) / RS_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void check_fp32(const ::brotensor::Tensor& t, const char* op,
                       const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32 (audio ops are FP32-only)");
    }
}

inline void check_args(const char* op, int N, int C, int L_in, int L_out,
                       int mode) {
    if (N < 0 || C < 0 || L_in < 0 || L_out < 0) {
        fail(op, "N, C, L_in, L_out must be non-negative");
    }
    if (mode != 0 && mode != 1) {
        fail(op, "mode must be 0 (nearest) or 1 (linear)");
    }
    if (L_out > 0 && L_in == 0) {
        fail(op, "L_in must be > 0 when L_out > 0");
    }
}

__device__ inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// One thread per output sample (n, c, dst).
__global__ void resample1d_forward_kernel(const float* __restrict__ X,
                                          float* __restrict__ Y,
                                          int N, int C, int L_in, int L_out,
                                          int mode, double scale) {
    const long long total = (long long)N * C * L_out;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int dst = static_cast<int>(idx % L_out);
        const long long t = idx / L_out;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);
        const long long xbase = ((long long)n * C + c) * L_in;
        const double src = (dst + 0.5) * scale - 0.5;
        if (mode == 0) {
            const int i = clampi(static_cast<int>(nearbyint(src)),
                                 0, L_in - 1);
            Y[idx] = X[xbase + i];
        } else {
            double s = src;
            if (s < 0.0) s = 0.0;
            if (s > L_in - 1) s = L_in - 1;
            const int x0 = static_cast<int>(floor(s));
            const int x1 = (x0 + 1 < L_in) ? x0 + 1 : L_in - 1;
            const float f = static_cast<float>(s - x0);
            Y[idx] = (1.0f - f) * X[xbase + x0] + f * X[xbase + x1];
        }
    }
}

// One thread per output gradient (n, c, dst); scatter-adds onto the input
// position(s) it sampled, with the forward weights.
__global__ void resample1d_backward_kernel(const float* __restrict__ dY,
                                           float* __restrict__ dX,
                                           int N, int C, int L_in, int L_out,
                                           int mode, double scale) {
    const long long total = (long long)N * C * L_out;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int dst = static_cast<int>(idx % L_out);
        const long long t = idx / L_out;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);
        const long long xbase = ((long long)n * C + c) * L_in;
        const double src = (dst + 0.5) * scale - 0.5;
        const float g = dY[idx];
        if (mode == 0) {
            const int i = clampi(static_cast<int>(nearbyint(src)),
                                 0, L_in - 1);
            atomicAdd(&dX[xbase + i], g);
        } else {
            double s = src;
            if (s < 0.0) s = 0.0;
            if (s > L_in - 1) s = L_in - 1;
            const int x0 = static_cast<int>(floor(s));
            const int x1 = (x0 + 1 < L_in) ? x0 + 1 : L_in - 1;
            const float f = static_cast<float>(s - x0);
            atomicAdd(&dX[xbase + x0], (1.0f - f) * g);
            atomicAdd(&dX[xbase + x1], f * g);
        }
    }
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════
//  Wrappers
// ════════════════════════════════════════════════════════════════════════════

void resample1d_forward(const ::brotensor::Tensor& X,
                        int N, int C, int L_in, int L_out, int mode,
                        ::brotensor::Tensor& Y) {
    check_fp32(X, "resample1d_forward", "X");
    check_args("resample1d_forward", N, C, L_in, L_out, mode);
    const int cols = C * L_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != ::brotensor::Dtype::FP32) {
        Y.resize(N, cols, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;
    const double scale = static_cast<double>(L_in) /
                         static_cast<double>(L_out);
    const long long total = (long long)N * C * L_out;
    resample1d_forward_kernel<<<rs_grid(total), RS_BLOCK>>>(
        static_cast<const float*>(X.data), static_cast<float*>(Y.data),
        N, C, L_in, L_out, mode, scale);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void resample1d_backward(const ::brotensor::Tensor& dY,
                         int N, int C, int L_in, int L_out, int mode,
                         ::brotensor::Tensor& dX) {
    check_fp32(dY, "resample1d_backward", "dY");
    check_args("resample1d_backward", N, C, L_in, L_out, mode);
    const int cols_in = C * L_in;
    if (dX.rows != N || dX.cols != cols_in ||
        dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(N, cols_in, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || cols_in == 0) return;
    // Adjoint: zero dX, then scatter each output gradient onto its taps.
    BROTENSOR_CUDA_CHECK(cudaMemset(
        dX.data, 0,
        static_cast<size_t>(N) * cols_in * sizeof(float)));
    if (L_out == 0) return;
    const double scale = static_cast<double>(L_in) /
                         static_cast<double>(L_out);
    const long long total = (long long)N * C * L_out;
    resample1d_backward_kernel<<<rs_grid(total), RS_BLOCK>>>(
        static_cast<const float*>(dY.data), static_cast<float*>(dX.data),
        N, C, L_in, L_out, mode, scale);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_resample1d(::brotensor::detail::OpsVTable& v) {
    v.resample1d_forward  = &resample1d_forward;
    v.resample1d_backward = &resample1d_backward;
}

} // namespace brotensor::detail::cuda
