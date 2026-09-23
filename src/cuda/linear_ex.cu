// linear_forward_batched_ex — 16-bit batched linear with a fused epilogue
// (store / residual accumulate / GeGLU) and optional split-K workspace.
// Runs on the mma.sync GEMM in gemm_mma.cu; shapes that kernel refuses
// (K or N not a multiple of 8, pre-sm_80) fall back to the WMMA linear plus
// an elementwise epilogue pass. CPU twin in src/cpu/packed_encoder.cpp.

#include <brotensor/detail/dispatch.h>
#include <brotensor/ops/linear.h>
#include <brotensor/tensor.h>

#include "detail/cuda_check.h"
#include "gemm_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor {
void* cuda_current_stream();
}

namespace brotensor::detail::cuda {

void linear_forward_batched_fp16_act(const ::brotensor::Tensor& W, const ::brotensor::Tensor* bias,
                                     const ::brotensor::Tensor& X_BD, int act,
                                     ::brotensor::Tensor& Y_BD);

namespace {

cudaStream_t cur_stream() { return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream()); }

[[noreturn]] void fail(const char* reason) {
    throw std::runtime_error(std::string("brotensor: linear_forward_batched_ex: ") + reason);
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// Fallback epilogue over a stored r (M, N): accumulate into Y (M, N) or
// GeGLU into Y (M, N/2).
template <typename T>
__global__ void epilogue_kernel(const T* __restrict__ r, T* __restrict__ y, long long count, int N, int epi) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (epi == 1) {
        y[i] = from_f<T>(to_f<T>(y[i]) + to_f<T>(r[i]));
    } else {
        const int half = N / 2;
        const long long m = i / half;
        const int j = static_cast<int>(i - m * half);
        const float a = to_f<T>(r[m * N + 2 * j]);
        const float g = to_f<T>(r[m * N + 2 * j + 1]);
        y[i] = from_f<T>(a * 0.5f * g * (1.0f + erff(g * 0.70710678118f)));
    }
}

template <typename T>
void run(const ::brotensor::Tensor& W, const ::brotensor::Tensor* bias, const ::brotensor::Tensor& X, int act,
         int epi, int acc_mode, ::brotensor::Tensor* ws, ::brotensor::Tensor& Y) {
    const int N = W.rows, K = W.cols, M = X.rows;
    float* wsp = nullptr;
    std::size_t ws_n = 0;
    if (ws) {
        const std::size_t need = mma_gemm::workspace_floats(M, N, K);
        if (need > 0) {
            if (ws->dtype != Dtype::FP32 || static_cast<std::size_t>(ws->rows) * ws->cols < need)
                ws->resize(1, static_cast<int>(need), Dtype::FP32);
            wsp = static_cast<float*>(ws->data);
            ws_n = static_cast<std::size_t>(ws->rows) * ws->cols;
        }
    }
    const T* bp = bias ? static_cast<const T*>(bias->data) : nullptr;
    if (mma_gemm::launch(static_cast<const T*>(X.data), static_cast<const T*>(W.data), static_cast<T*>(Y.data), M,
                         N, K, bp, act, epi, wsp, ws_n, cur_stream(), acc_mode))
        return;
    if (epi == 0) {
        detail::cuda::linear_forward_batched_fp16_act(W, bias, X, act, Y);
        return;
    }
    ::brotensor::Tensor r = ::brotensor::Tensor::empty_on(W.device, M, N, W.dtype);
    detail::cuda::linear_forward_batched_fp16_act(W, bias, X, act, r);
    const long long count = epi == 1 ? static_cast<long long>(M) * N : static_cast<long long>(M) * (N / 2);
    if (count == 0) return;
    const int threads = 256;
    const unsigned blocks = static_cast<unsigned>((count + threads - 1) / threads);
    epilogue_kernel<T><<<blocks, threads, 0, cur_stream()>>>(static_cast<const T*>(r.data), static_cast<T*>(Y.data),
                                                             count, N, epi);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void linear_forward_batched_ex(const ::brotensor::Tensor& W, const ::brotensor::Tensor* bias,
                               const ::brotensor::Tensor& X, int act, int epilogue_flags,
                               ::brotensor::Tensor* workspace, ::brotensor::Tensor& Y) {
    const int acc_mode = (epilogue_flags & kLinearEpiFastAccum) ? mma_gemm::kAccHybrid : mma_gemm::kAccF32;
    const int epilogue = epilogue_flags & ~kLinearEpiFastAccum;
    if (W.dtype != Dtype::FP16 && W.dtype != Dtype::BF16) fail("W must be FP16 or BF16 on CUDA");
    if (X.dtype != W.dtype) fail("X must share W's dtype");
    if (bias && bias->dtype != W.dtype) fail("bias must share W's dtype");
    const int N = W.rows, K = W.cols, M = X.rows;
    if (X.cols != K) fail("X.cols must equal W.cols");
    if (bias && static_cast<long long>(bias->rows) * bias->cols != N) fail("bias size must equal W.rows");
    if (epilogue < 0 || epilogue > 2) fail("unknown epilogue");
    if (epilogue == 2 && (act != 0 || N % 2 != 0)) fail("geglu needs act 0 and even W.rows");
    const int out_cols = epilogue == 2 ? N / 2 : N;
    if (epilogue == 1) {
        if (Y.rows != M || Y.cols != N || Y.dtype != W.dtype) fail("accumulate needs Y (B, out) in W's dtype");
    } else if (Y.rows != M || Y.cols != out_cols || Y.dtype != W.dtype) {
        Y.resize(M, out_cols, W.dtype);
    }
    if (M == 0 || N == 0) return;
    if (W.dtype == Dtype::FP16)
        run<__half>(W, bias, X, act, epilogue, acc_mode, workspace, Y);
    else
        run<__nv_bfloat16>(W, bias, X, act, epilogue, acc_mode, workspace, Y);
}

void fill_cuda_vtable_linear_ex(::brotensor::detail::OpsVTable& v) {
    v.linear_forward_batched_ex = &linear_forward_batched_ex;
}

}  // namespace brotensor::detail::cuda
