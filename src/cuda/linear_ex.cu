// linear_forward_batched_ex — 16-bit batched linear with a fused epilogue
// (store / residual accumulate / GeGLU / SwiGLU) and optional split-K
// workspace. One to eight rows take the skinny GEMV below (decode); 9..64 rows
// the weight-streaming tensor-core kernel in gemm_skinny.cu (every epilogue
// fused); wider batches run on the mma.sync GEMM in gemm_mma.cu; shapes that kernel refuses
// (K or N not a multiple of 8, pre-sm_80) and SwiGLU fall back to a stored
// result plus an elementwise epilogue pass. CPU twin in
// src/cpu/packed_encoder.cpp.

#include <cstdint>
#include <type_traits>

#include <brotensor/detail/dispatch.h>
#include <brotensor/ops/linear.h>
#include <brotensor/tensor.h>

#include "detail/cuda_check.h"
#include "gemm_mma.cuh"
#include "gemm_skinny.cuh"

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

// Fallback epilogue over a stored r (M, N): accumulate into Y (M, N), GeGLU
// or SwiGLU into Y (M, N/2).
template <typename T>
__global__ void epilogue_kernel(const T* __restrict__ r, T* __restrict__ y, long long count, int N, int epi) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (epi == 1) {
        y[i] = from_f<T>(to_f<T>(y[i]) + to_f<T>(r[i]));
    } else if (epi == 3) {
        const int half = N / 2;
        const long long m = i / half;
        const int j = static_cast<int>(i - m * half);
        const float g = to_f<T>(r[m * N + j]);
        const float u = to_f<T>(r[m * N + half + j]);
        y[i] = from_f<T>(g / (1.0f + __expf(-g)) * u);
    } else {
        const int half = N / 2;
        const long long m = i / half;
        const int j = static_cast<int>(i - m * half);
        const float a = to_f<T>(r[m * N + 2 * j]);
        const float g = to_f<T>(r[m * N + 2 * j + 1]);
        y[i] = from_f<T>(a * 0.5f * g * (1.0f + erff(g * 0.70710678118f)));
    }
}

// ── Skinny rows (M <= 8): one weight pass for every row ─────────────────────
// Decode-shaped linears are weight-bandwidth bound. One CTA per output column
// reads its weight row (two rows for a gated epilogue) once with 16-byte
// loads and dots it against all M input rows, FP32 accumulation, the epilogue
// fused into the store — where the tile GEMM at M = 1..8 wastes most of each
// tile and re-reads nothing better, and the block-per-(row, batch) GEMV reads
// the weights M times.
constexpr int GV_BLOCK = 128;

template <typename T2>
struct __align__(16) Quad { T2 a, b, c, d; };

__device__ __forceinline__ float lo(__half2 v) { return __low2float(v); }
__device__ __forceinline__ float hi(__half2 v) { return __high2float(v); }
__device__ __forceinline__ float lo(__nv_bfloat162 v) { return __low2float(v); }
__device__ __forceinline__ float hi(__nv_bfloat162 v) { return __high2float(v); }

template <typename T2>
__device__ __forceinline__ float dot8(const Quad<T2>& w, const Quad<T2>& x) {
    return lo(w.a) * lo(x.a) + hi(w.a) * hi(x.a) + lo(w.b) * lo(x.b) + hi(w.b) * hi(x.b) +
           lo(w.c) * lo(x.c) + hi(w.c) * hi(x.c) + lo(w.d) * lo(x.d) + hi(w.d) * hi(x.d);
}

__device__ __forceinline__ float act_f(int act, float v) {
    switch (act) {
        case 1: return v > 0.0f ? v : 0.0f;
        case 2: return 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
        case 3: return 0.5f * v * (1.0f + erff(v * 0.70710678118f));
        case 4: return v / (1.0f + __expf(-v));
        case 5: return v / (1.0f + __expf(-1.702f * v));
        default: return v;
    }
}

// EPI 0 store, 1 accumulate, 2 GeGLU (rows 2j, 2j+1), 3 SwiGLU (rows j, N/2+j).
template <typename T, typename T2, int MR, int EPI>
__global__ void __launch_bounds__(GV_BLOCK)
gemv_ex_kernel(const T* __restrict__ W, const T* __restrict__ bias, const T* __restrict__ X, T* __restrict__ Y,
               int N, int K, int act) {
    constexpr int NW = EPI >= 2 ? 2 : 1;
    const int j = blockIdx.x;
    const int tid = threadIdx.x;
    const int out_cols = EPI >= 2 ? N / 2 : N;
    int rows[NW];
    rows[0] = EPI == 2 ? 2 * j : j;
    if constexpr (NW == 2) rows[1] = EPI == 2 ? 2 * j + 1 : N / 2 + j;
    const int n8 = K / 8;
    float acc[NW][MR];
#pragma unroll
    for (int w = 0; w < NW; ++w)
#pragma unroll
        for (int m = 0; m < MR; ++m) acc[w][m] = 0.0f;
    const Quad<T2>* x8 = reinterpret_cast<const Quad<T2>*>(X);
    const Quad<T2>* w8[NW];
#pragma unroll
    for (int w = 0; w < NW; ++w) w8[w] = reinterpret_cast<const Quad<T2>*>(W + static_cast<size_t>(rows[w]) * K);
    int k = tid;
    // Two quads per weight row in flight per iteration.
    for (; k + GV_BLOCK < n8; k += 2 * GV_BLOCK) {
        Quad<T2> wq[NW][2];
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            wq[w][0] = w8[w][k];
            wq[w][1] = w8[w][k + GV_BLOCK];
        }
#pragma unroll
        for (int m = 0; m < MR; ++m) {
            const Quad<T2> x0 = x8[static_cast<size_t>(m) * n8 + k];
            const Quad<T2> x1 = x8[static_cast<size_t>(m) * n8 + k + GV_BLOCK];
#pragma unroll
            for (int w = 0; w < NW; ++w) acc[w][m] += dot8(wq[w][0], x0) + dot8(wq[w][1], x1);
        }
    }
    for (; k < n8; k += GV_BLOCK) {
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            const Quad<T2> wq = w8[w][k];
#pragma unroll
            for (int m = 0; m < MR; ++m) acc[w][m] += dot8(wq, x8[static_cast<size_t>(m) * n8 + k]);
        }
    }
    __shared__ float red[NW * MR][GV_BLOCK / 32];
#pragma unroll
    for (int w = 0; w < NW; ++w)
#pragma unroll
        for (int m = 0; m < MR; ++m) {
            float v = acc[w][m];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
            if ((tid & 31) == 0) red[w * MR + m][tid >> 5] = v;
        }
    __syncthreads();
    if (tid < MR) {
        const int m = tid;
        float s[NW];
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            float t = 0.0f;
#pragma unroll
            for (int q = 0; q < GV_BLOCK / 32; ++q) t += red[w * MR + m][q];
            if (bias) t += to_f<T>(bias[rows[w]]);
            s[w] = t;
        }
        T* y = Y + static_cast<size_t>(m) * out_cols + j;
        if constexpr (EPI == 0) {
            *y = from_f<T>(act_f(act, s[0]));
        } else if constexpr (EPI == 1) {
            *y = from_f<T>(to_f<T>(*y) + act_f(act, s[0]));
        } else if constexpr (EPI == 2) {
            *y = from_f<T>(s[0] * 0.5f * s[1] * (1.0f + erff(s[1] * 0.70710678118f)));
        } else {
            *y = from_f<T>(s[0] / (1.0f + __expf(-s[0])) * s[NW - 1]);
        }
    }
}

template <typename T, typename T2, int MR>
void gemv_launch_m(const T* W, const T* b, const T* X, T* Y, int N, int K, int act, int epi, cudaStream_t st) {
    const unsigned grid = static_cast<unsigned>(epi >= 2 ? N / 2 : N);
    switch (epi) {
        case 0: gemv_ex_kernel<T, T2, MR, 0><<<grid, GV_BLOCK, 0, st>>>(W, b, X, Y, N, K, act); break;
        case 1: gemv_ex_kernel<T, T2, MR, 1><<<grid, GV_BLOCK, 0, st>>>(W, b, X, Y, N, K, act); break;
        case 2: gemv_ex_kernel<T, T2, MR, 2><<<grid, GV_BLOCK, 0, st>>>(W, b, X, Y, N, K, act); break;
        default: gemv_ex_kernel<T, T2, MR, 3><<<grid, GV_BLOCK, 0, st>>>(W, b, X, Y, N, K, act); break;
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
bool gemv_ex(const T* W, const T* b, const T* X, T* Y, int M, int N, int K, int act, int epi, cudaStream_t st) {
    if (M < 1 || M > 8 || (K & 7)) return false;
    auto al = [](const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; };
    if (!al(W) || !al(X)) return false;
    using T2 = std::conditional_t<std::is_same_v<T, __half>, __half2, __nv_bfloat162>;
    switch (M) {
        case 1: gemv_launch_m<T, T2, 1>(W, b, X, Y, N, K, act, epi, st); break;
        case 2: gemv_launch_m<T, T2, 2>(W, b, X, Y, N, K, act, epi, st); break;
        case 3: gemv_launch_m<T, T2, 3>(W, b, X, Y, N, K, act, epi, st); break;
        case 4: gemv_launch_m<T, T2, 4>(W, b, X, Y, N, K, act, epi, st); break;
        case 5: gemv_launch_m<T, T2, 5>(W, b, X, Y, N, K, act, epi, st); break;
        case 6: gemv_launch_m<T, T2, 6>(W, b, X, Y, N, K, act, epi, st); break;
        case 7: gemv_launch_m<T, T2, 7>(W, b, X, Y, N, K, act, epi, st); break;
        default: gemv_launch_m<T, T2, 8>(W, b, X, Y, N, K, act, epi, st); break;
    }
    return true;
}

template <typename T>
void run(const ::brotensor::Tensor& W, const ::brotensor::Tensor* bias, const ::brotensor::Tensor& X, int act,
         int epi, int acc_mode, ::brotensor::Tensor* ws, ::brotensor::Tensor& Y) {
    const int N = W.rows, K = W.cols, M = X.rows;
    const T* bp0 = bias ? static_cast<const T*>(bias->data) : nullptr;
    if (gemv_ex<T>(static_cast<const T*>(W.data), bp0, static_cast<const T*>(X.data), static_cast<T*>(Y.data), M, N,
                   K, act, epi, cur_stream()))
        return;
    const T* bp = bias ? static_cast<const T*>(bias->data) : nullptr;
    if (M <= 64) {
        float* sp = nullptr;
        std::size_t sn = 0;
        const std::size_t need = gemm_skinny::workspace_floats(M, N, K, epi);
        if (ws && need > 0) {
            if (ws->dtype != Dtype::FP32 || static_cast<std::size_t>(ws->rows) * ws->cols < need)
                ws->resize(1, static_cast<int>(need), Dtype::FP32);
            sp = static_cast<float*>(ws->data);
            sn = static_cast<std::size_t>(ws->rows) * ws->cols;
        }
        if (gemm_skinny::launch(static_cast<const T*>(X.data), static_cast<const T*>(W.data),
                                static_cast<T*>(Y.data), M, N, K, bp, act, epi, sp, sn, cur_stream()))
            return;
    }
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
    // The tile GEMM has no SwiGLU epilogue: wider batches store r and run the
    // epilogue pass below.
    if (epi != 3 && mma_gemm::launch(static_cast<const T*>(X.data), static_cast<const T*>(W.data),
                                     static_cast<T*>(Y.data), M, N, K, bp, act, epi, wsp, ws_n, cur_stream(),
                                     acc_mode))
        return;
    if (epi == 0) {
        detail::cuda::linear_forward_batched_fp16_act(W, bias, X, act, Y);
        return;
    }
    ::brotensor::Tensor r = ::brotensor::Tensor::empty_on(W.device, M, N, W.dtype);
    if (epi != 3 || !mma_gemm::launch(static_cast<const T*>(X.data), static_cast<const T*>(W.data),
                                      static_cast<T*>(r.data), M, N, K, bp, act, 0, wsp, ws_n, cur_stream(),
                                      acc_mode))
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
    if (epilogue < 0 || epilogue > 3) fail("unknown epilogue");
    if (epilogue >= 2 && (act != 0 || N % 2 != 0)) fail("geglu/swiglu needs act 0 and even W.rows");
    const int out_cols = epilogue >= 2 ? N / 2 : N;
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
