// Weight-only quantised linear: quantize_rows + linear_forward_batched_wq_ex
// (INT8 or FP8 E4M3 weights, one FP32 scale per output row, FP16/BF16
// activations). Decode GEMVs are weight-bandwidth bound, so a 1-byte weight
// halves their floor against BF16. One warp per output column (two weight
// rows for a gated epilogue) streams its row with 16-byte loads, decodes in
// registers and dots it against up to eight input rows at once; the row scale
// multiplies the FP32 sum, then bias, activation and the epilogue (store,
// accumulate, GeGLU, SwiGLU) as linear_forward_batched_ex does them. More
// than eight rows run in slices of eight (each slice re-reads the weight).
// CPU twin in src/cpu/linear_wq.cpp.

#include <cstdint>
#include <type_traits>

#include <brotensor/detail/dispatch.h>
#include <brotensor/ops/linear.h>
#include <brotensor/tensor.h>

#include "detail/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor {
void* cuda_current_stream();
}

namespace brotensor::detail::cuda {

namespace {

cudaStream_t cur_stream() { return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream()); }

[[noreturn]] void fail(const char* op, const char* reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

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

// ── decode ──────────────────────────────────────────────────────────────────
// FMT 0 = INT8, 1 = FP8 E4M3. Four weights of one 32-bit word to floats.
template <int FMT>
__device__ __forceinline__ void dec4(uint32_t u, float* f) {
    if constexpr (FMT == 0) {
        // Bias each byte by 128 and drop it into the mantissa of 2^23.
        const uint32_t v = u ^ 0x80808080u;
        f[0] = __uint_as_float(__byte_perm(v, 0x4B000000u, 0x7650)) - 8388736.0f;
        f[1] = __uint_as_float(__byte_perm(v, 0x4B000000u, 0x7651)) - 8388736.0f;
        f[2] = __uint_as_float(__byte_perm(v, 0x4B000000u, 0x7652)) - 8388736.0f;
        f[3] = __uint_as_float(__byte_perm(v, 0x4B000000u, 0x7653)) - 8388736.0f;
    } else {
        const __half2_raw a = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u & 0xffffu), __NV_E4M3);
        const __half2_raw b = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u >> 16), __NV_E4M3);
        const float2 fa = __half22float2(*reinterpret_cast<const __half2*>(&a));
        const float2 fb = __half22float2(*reinterpret_cast<const __half2*>(&b));
        f[0] = fa.x;
        f[1] = fa.y;
        f[2] = fb.x;
        f[3] = fb.y;
    }
}

// Eight activations of one 16-byte load to floats.
template <typename T>
__device__ __forceinline__ void dec8x(const uint4& q, float* f) {
    const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            f[2 * i] = __uint_as_float(w[i] << 16);
            f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
        } else {
            const float2 h = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
            f[2 * i] = h.x;
            f[2 * i + 1] = h.y;
        }
    }
}

constexpr int WQ_WARPS = 4;

// EPI 0 store, 1 accumulate, 2 GeGLU (rows 2j, 2j+1), 3 SwiGLU (rows j, N/2+j).
template <int FMT, typename T, int MR, int EPI>
__global__ void __launch_bounds__(WQ_WARPS * 32)
gemv_wq_kernel(const uint8_t* __restrict__ W, const float* __restrict__ scales, const T* __restrict__ bias,
               const T* __restrict__ X, T* __restrict__ Y, int N, int K, int act) {
    constexpr int NW = EPI >= 2 ? 2 : 1;
    const int lane = threadIdx.x & 31;
    const int out_cols = EPI >= 2 ? N / 2 : N;
    const int j = blockIdx.x * WQ_WARPS + (threadIdx.x >> 5);
    if (j >= out_cols) return;
    int rows[NW];
    rows[0] = EPI == 2 ? 2 * j : j;
    if constexpr (NW == 2) rows[1] = EPI == 2 ? 2 * j + 1 : N / 2 + j;
    const int n16 = K / 16, k8 = K / 8;
    float acc[NW][MR];
#pragma unroll
    for (int w = 0; w < NW; ++w)
#pragma unroll
        for (int m = 0; m < MR; ++m) acc[w][m] = 0.0f;
    const uint4* wrow[NW];
#pragma unroll
    for (int w = 0; w < NW; ++w) wrow[w] = reinterpret_cast<const uint4*>(W + static_cast<size_t>(rows[w]) * K);
    const uint4* x8 = reinterpret_cast<const uint4*>(X);

    auto step = [&](const uint4 (&wq)[NW], int c) {
        float wf[NW][16];
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            dec4<FMT>(wq[w].x, wf[w]);
            dec4<FMT>(wq[w].y, wf[w] + 4);
            dec4<FMT>(wq[w].z, wf[w] + 8);
            dec4<FMT>(wq[w].w, wf[w] + 12);
        }
#pragma unroll
        for (int m = 0; m < MR; ++m) {
            float xf[16];
            dec8x<T>(__ldg(x8 + static_cast<size_t>(m) * k8 + 2 * c), xf);
            dec8x<T>(__ldg(x8 + static_cast<size_t>(m) * k8 + 2 * c + 1), xf + 8);
#pragma unroll
            for (int w = 0; w < NW; ++w) {
                float s = 0.0f;
#pragma unroll
                for (int e = 0; e < 16; ++e) s = fmaf(wf[w][e], xf[e], s);
                acc[w][m] += s;
            }
        }
    };

    int c = lane;
    // Two 16-byte weight loads per row in flight per iteration.
    for (; c + 32 < n16; c += 64) {
        uint4 a[NW], b[NW];
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            a[w] = __ldcs(wrow[w] + c);
            b[w] = __ldcs(wrow[w] + c + 32);
        }
        step(a, c);
        step(b, c + 32);
    }
    for (; c < n16; c += 32) {
        uint4 a[NW];
#pragma unroll
        for (int w = 0; w < NW; ++w) a[w] = __ldcs(wrow[w] + c);
        step(a, c);
    }
#pragma unroll
    for (int w = 0; w < NW; ++w)
#pragma unroll
        for (int m = 0; m < MR; ++m) {
            float v = acc[w][m];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
            acc[w][m] = v;
        }
    if (lane >= MR) return;
    // Lane m stores row m (the sums are warp-uniform after the xor reduce).
    float s[NW];
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        float t = 0.0f;
#pragma unroll
        for (int m = 0; m < MR; ++m)
            if (m == lane) t = acc[w][m];
        t *= scales[rows[w]];
        if (bias) t += to_f<T>(bias[rows[w]]);
        s[w] = t;
    }
    T* y = Y + static_cast<size_t>(lane) * out_cols + j;
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

// Any shape (K not a multiple of 16, unaligned pointers): one thread per
// output element, scalar loads.
template <int FMT, typename T>
__global__ void wq_scalar_kernel(const uint8_t* __restrict__ W, const float* __restrict__ scales,
                                 const T* __restrict__ bias, const T* __restrict__ X, T* __restrict__ Y, int M, int N,
                                 int K, int act, int epi) {
    const int out_cols = epi >= 2 ? N / 2 : N;
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(M) * out_cols) return;
    const int m = static_cast<int>(i / out_cols), j = static_cast<int>(i % out_cols);
    const int nw = epi >= 2 ? 2 : 1;
    float s[2] = {0.0f, 0.0f};
    for (int w = 0; w < nw; ++w) {
        const int row = epi == 2 ? 2 * j + w : (epi == 3 ? (w ? N / 2 + j : j) : j);
        const uint8_t* wr = W + static_cast<size_t>(row) * K;
        const T* xr = X + static_cast<size_t>(m) * K;
        float a = 0.0f;
        for (int k = 0; k < K; ++k) {
            float f[4];
            dec4<FMT>(wr[k], f);   // only byte 0 is used
            a = fmaf(f[0], to_f<T>(xr[k]), a);
        }
        a *= scales[row];
        if (bias) a += to_f<T>(bias[row]);
        s[w] = a;
    }
    T* y = Y + static_cast<size_t>(m) * out_cols + j;
    if (epi == 0) *y = from_f<T>(act_f(act, s[0]));
    else if (epi == 1) *y = from_f<T>(to_f<T>(*y) + act_f(act, s[0]));
    else if (epi == 2) *y = from_f<T>(s[0] * 0.5f * s[1] * (1.0f + erff(s[1] * 0.70710678118f)));
    else *y = from_f<T>(s[0] / (1.0f + __expf(-s[0])) * s[1]);
}

template <int FMT, typename T, int MR>
void launch_m(const uint8_t* W, const float* sc, const T* b, const T* X, T* Y, int N, int K, int act, int epi,
              cudaStream_t st) {
    const int out_cols = epi >= 2 ? N / 2 : N;
    const unsigned grid = static_cast<unsigned>((out_cols + WQ_WARPS - 1) / WQ_WARPS);
    const unsigned blk = WQ_WARPS * 32;
    switch (epi) {
        case 0: gemv_wq_kernel<FMT, T, MR, 0><<<grid, blk, 0, st>>>(W, sc, b, X, Y, N, K, act); break;
        case 1: gemv_wq_kernel<FMT, T, MR, 1><<<grid, blk, 0, st>>>(W, sc, b, X, Y, N, K, act); break;
        case 2: gemv_wq_kernel<FMT, T, MR, 2><<<grid, blk, 0, st>>>(W, sc, b, X, Y, N, K, act); break;
        default: gemv_wq_kernel<FMT, T, MR, 3><<<grid, blk, 0, st>>>(W, sc, b, X, Y, N, K, act); break;
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

template <int FMT, typename T>
void run(const uint8_t* W, const float* sc, const T* b, const T* X, T* Y, int M, int N, int K, int act, int epi) {
    cudaStream_t st = cur_stream();
    const int out_cols = epi >= 2 ? N / 2 : N;
    auto al = [](const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; };
    if ((K & 15) || !al(W) || !al(X)) {
        const long long count = static_cast<long long>(M) * out_cols;
        const unsigned blocks = static_cast<unsigned>((count + 255) / 256);
        wq_scalar_kernel<FMT, T><<<blocks, 256, 0, st>>>(W, sc, b, X, Y, M, N, K, act, epi);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }
    for (int m0 = 0; m0 < M; m0 += 8) {
        const int mr = M - m0 < 8 ? M - m0 : 8;
        const T* x = X + static_cast<size_t>(m0) * K;
        T* y = Y + static_cast<size_t>(m0) * out_cols;
        switch (mr) {
            case 1: launch_m<FMT, T, 1>(W, sc, b, x, y, N, K, act, epi, st); break;
            case 2: launch_m<FMT, T, 2>(W, sc, b, x, y, N, K, act, epi, st); break;
            case 3: launch_m<FMT, T, 3>(W, sc, b, x, y, N, K, act, epi, st); break;
            case 4: launch_m<FMT, T, 4>(W, sc, b, x, y, N, K, act, epi, st); break;
            case 5: launch_m<FMT, T, 5>(W, sc, b, x, y, N, K, act, epi, st); break;
            case 6: launch_m<FMT, T, 6>(W, sc, b, x, y, N, K, act, epi, st); break;
            case 7: launch_m<FMT, T, 7>(W, sc, b, x, y, N, K, act, epi, st); break;
            default: launch_m<FMT, T, 8>(W, sc, b, x, y, N, K, act, epi, st); break;
        }
    }
}

// ── quantize_rows ──────────────────────────────────────────────────────────
// One CTA per row: amax, scale = amax / qmax, then each element / scale
// rounded to the format (INT8 round-half-even clamped to +-127; E4M3
// round-to-nearest-even, saturating at +-448).
template <typename T>
__global__ void quantize_rows_kernel(const T* __restrict__ W, int K, int fmt, uint8_t* __restrict__ Q,
                                     float* __restrict__ scales) {
    const int r = blockIdx.x;
    const T* w = W + static_cast<size_t>(r) * K;
    float mx = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) mx = fmaxf(mx, fabsf(to_f<T>(w[k])));
    __shared__ float red[32];
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float amax = red[0];
    const float qmax = fmt == 0 ? 127.0f : 448.0f;
    const float scale = amax / qmax;
    if (threadIdx.x == 0) scales[r] = scale;
    uint8_t* q = Q + static_cast<size_t>(r) * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const float v = scale > 0.0f ? to_f<T>(w[k]) / scale : 0.0f;
        if (fmt == 0) {
            const float c = fminf(fmaxf(rintf(v), -127.0f), 127.0f);
            q[k] = static_cast<uint8_t>(static_cast<int8_t>(c));
        } else {
            q[k] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
        }
    }
}

}  // namespace

void linear_forward_batched_wq_ex(const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& scales,
                                  const ::brotensor::Tensor* bias, const ::brotensor::Tensor& X, int act,
                                  int epilogue_flags, ::brotensor::Tensor& Y) {
    constexpr const char* op = "linear_forward_batched_wq_ex";
    const int epilogue = epilogue_flags & ~kLinearEpiFastAccum;
    if (Wq.dtype != Dtype::INT8 && Wq.dtype != Dtype::FP8_E4M3) fail(op, "Wq must be INT8 or FP8_E4M3");
    if (scales.dtype != Dtype::FP32 || static_cast<long long>(scales.rows) * scales.cols != Wq.rows)
        fail(op, "scales must be (out,1) FP32");
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) fail(op, "X must be FP16 or BF16 on CUDA");
    if (bias && bias->dtype != X.dtype) fail(op, "bias must share X's dtype");
    const int N = Wq.rows, K = Wq.cols, M = X.rows;
    if (X.cols != K) fail(op, "X.cols must equal Wq.cols");
    if (bias && static_cast<long long>(bias->rows) * bias->cols != N) fail(op, "bias size must equal Wq.rows");
    if (epilogue < 0 || epilogue > 3) fail(op, "unknown epilogue");
    if (epilogue >= 2 && (act != 0 || N % 2 != 0)) fail(op, "geglu/swiglu needs act 0 and even Wq.rows");
    const int out_cols = epilogue >= 2 ? N / 2 : N;
    if (epilogue == 1) {
        if (Y.rows != M || Y.cols != N || Y.dtype != X.dtype) fail(op, "accumulate needs Y (B, out) in X's dtype");
    } else if (Y.rows != M || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(M, out_cols, X.dtype);
    }
    if (M == 0 || N == 0) return;
    const auto* w = static_cast<const uint8_t*>(Wq.data);
    const auto* sc = static_cast<const float*>(scales.data);
    const bool fp8 = Wq.dtype == Dtype::FP8_E4M3;
    if (X.dtype == Dtype::BF16) {
        using T = __nv_bfloat16;
        const T* b = bias ? static_cast<const T*>(bias->data) : nullptr;
        if (fp8) run<1, T>(w, sc, b, static_cast<const T*>(X.data), static_cast<T*>(Y.data), M, N, K, act, epilogue);
        else run<0, T>(w, sc, b, static_cast<const T*>(X.data), static_cast<T*>(Y.data), M, N, K, act, epilogue);
    } else {
        using T = __half;
        const T* b = bias ? static_cast<const T*>(bias->data) : nullptr;
        if (fp8) run<1, T>(w, sc, b, static_cast<const T*>(X.data), static_cast<T*>(Y.data), M, N, K, act, epilogue);
        else run<0, T>(w, sc, b, static_cast<const T*>(X.data), static_cast<T*>(Y.data), M, N, K, act, epilogue);
    }
}

void quantize_rows(const ::brotensor::Tensor& W, Dtype fmt, ::brotensor::Tensor& Wq, ::brotensor::Tensor& scales) {
    constexpr const char* op = "quantize_rows";
    if (fmt != Dtype::INT8 && fmt != Dtype::FP8_E4M3) fail(op, "fmt must be INT8 or FP8_E4M3");
    if (W.dtype != Dtype::FP16 && W.dtype != Dtype::BF16 && W.dtype != Dtype::FP32)
        fail(op, "W must be FP16, BF16 or FP32");
    const int N = W.rows, K = W.cols;
    if (Wq.rows != N || Wq.cols != K || Wq.dtype != fmt) Wq.resize(N, K, fmt);
    if (scales.rows != N || scales.cols != 1 || scales.dtype != Dtype::FP32) scales.resize(N, 1, Dtype::FP32);
    if (N == 0 || K == 0) return;
    const int f = fmt == Dtype::INT8 ? 0 : 1;
    auto* q = static_cast<uint8_t*>(Wq.data);
    auto* s = static_cast<float*>(scales.data);
    cudaStream_t st = cur_stream();
    const unsigned grid = static_cast<unsigned>(N);
    if (W.dtype == Dtype::BF16)
        quantize_rows_kernel<__nv_bfloat16><<<grid, 256, 0, st>>>(static_cast<const __nv_bfloat16*>(W.data), K, f, q, s);
    else if (W.dtype == Dtype::FP16)
        quantize_rows_kernel<__half><<<grid, 256, 0, st>>>(static_cast<const __half*>(W.data), K, f, q, s);
    else
        quantize_rows_kernel<float><<<grid, 256, 0, st>>>(static_cast<const float*>(W.data), K, f, q, s);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void fill_cuda_vtable_linear_wq(::brotensor::detail::OpsVTable& v) {
    v.linear_forward_batched_wq_ex = &linear_forward_batched_wq_ex;
    v.quantize_rows = &quantize_rows;
}

}  // namespace brotensor::detail::cuda
