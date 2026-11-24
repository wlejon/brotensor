#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstddef>
#include <stdexcept>

// BROTENSOR_CUDA_CHECK macro lives here.
#include "detail/cuda_check.h"

namespace brotensor {

// Defined in runtime.cu. Thread-local current CUDA stream as opaque
// void*. Kept as a forward declaration here so this TU does not need a public
// header for it.
void* cuda_current_stream();

namespace conv2d_wmma_internal {
// Defined in conv2d_wmma.cu. Returns true iff it consumed the call; returns
// false if the shape isn't on the WMMA fast path and the caller should fall
// back to the naive direct-conv kernel below.
bool launch_conv2d_implicit_gemm_wmma(
        const __half* X, const __half* Wt, const __half* bias, __half* Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int H_out, int W_out);
// BF16 twin of the WMMA forward path. RTX 4090 (sm_89) supports BF16 WMMA
// fragments; same shape gating as the FP16 entry point.
bool launch_conv2d_implicit_gemm_wmma_bf16(
        const __nv_bfloat16* X, const __nv_bfloat16* Wt,
        const __nv_bfloat16* bias, __nv_bfloat16* Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int H_out, int W_out);
}

namespace detail::cuda::matmul_rm_internal {
// Defined in matmul.cu. Row-major C(M,N) = A(M,K) @ B(K,N) on tensor cores.
// Returns false without launching when the shape falls outside the WMMA path's
// gating (K or N not 8-aligned, K < 16, tiny M*N), so the caller can fall back.
bool launch_matmul_rm_wmma(const __half* A, const __half* B, __half* C,
                           int M, int N, int K);
bool launch_matmul_rm_wmma_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B,
                                __nv_bfloat16* C, int M, int N, int K);
// CTAs one such launch would occupy, for the batch-loop trade-off below.
int rm_wmma_grid_ctas(int M, int N);
}

namespace {

constexpr int CONV_BLOCK = 256;

template <typename T>
__device__ inline float load_f32(const T* p);
template <> __device__ inline float load_f32<float>(const float* p)   { return *p; }
template <> __device__ inline float load_f32<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float load_f32<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }

template <typename T>
__device__ inline void store_f32(T* p, float v);
template <> __device__ inline void store_f32<float>(float* p, float v)   { *p = v; }
template <> __device__ inline void store_f32<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void store_f32<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// One BLOCK per (c_out, c_in, kh, kw) element of dWt. Threads stride over
// the M = N*H_out*W_out reduction domain and a shared-memory tree reduce
// folds the partials; thread 0 writes the FP32 scratch slot. No atomics.
// This is the small-weight-grid path: for depthwise / pointwise convs the
// element count is only hundreds-to-thousands, so the thread-per-element
// kernel below would put the whole N*spatial reduction in a serial loop on
// a tiny fraction of the GPU (and its time scales linearly with N).
template <typename T>
__global__ void conv2d_backward_weight_block_kernel(
        const T* __restrict__ X,
        const T* __restrict__ dY,
        float* __restrict__ dWt_scratch,    // FP32, size C_out*Cg_in*kH*kW
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int groups, int Cg_in, int Cg_out) {
    const int idx = blockIdx.x;   // weight element
    const int tid = threadIdx.x;

    // Unflatten idx → (c_out, c_in_local, kh, kw) in OIHW layout
    // (I-dim sized as Cg_in for grouped conv).
    const int kw = idx % kW;
    int t = idx / kW;
    const int kh = t % kH;
    t /= kH;
    const int c_in_local = t % Cg_in;
    const int c_out      = t / Cg_in;

    // Absolute input channel for this (c_out, c_in_local).
    const int g = c_out / Cg_out;
    const int c_in = g * Cg_in + c_in_local;

    const int spatial = H_out * W_out;
    const int M = N * spatial;

    float acc = 0.0f;
    for (int m = tid; m < M; m += blockDim.x) {
        const int n  = m / spatial;
        const int sp = m - n * spatial;
        const int i_out = sp / W_out;
        const int j_out = sp - i_out * W_out;
        const int in_h = i_out * stride_h - pad_h + kh * dil_h;
        if (in_h < 0 || in_h >= H) continue;
        const int in_w = j_out * stride_w - pad_w + kw * dil_w;
        if (in_w < 0 || in_w >= W) continue;
        const int x_idx  = ((n * C_in  + c_in)  * H     + in_h)  * W     + in_w;
        const int dy_idx = ((n * C_out + c_out) * H_out + i_out) * W_out + j_out;
        acc += load_f32<T>(&dY[dy_idx]) * load_f32<T>(&X[x_idx]);
    }

    __shared__ float s_acc[CONV_BLOCK];
    s_acc[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_acc[tid] += s_acc[tid + stride];
        __syncthreads();
    }
    if (tid == 0) dWt_scratch[idx] = s_acc[0];
}

// One thread per (c_out, c_in, kh, kw) element of dWt. Iterates (n, i_out,
// j_out) and accumulates into an FP32 scratch slot. No atomics. The FP32
// scratch is folded into the caller's dWt (storage-dtype-dispatched).
// Kept as the large-weight-grid path: with ≳64K elements the grid already
// fills the GPU and the per-thread loop is short.
template <typename T>
__global__ void conv2d_backward_weight_kernel(
        const T* __restrict__ X,
        const T* __restrict__ dY,
        float* __restrict__ dWt_scratch,    // FP32, size C_out*Cg_in*kH*kW
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int groups, int Cg_in, int Cg_out,
        int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Unflatten idx → (c_out, c_in_local, kh, kw) in OIHW layout
    // (I-dim sized as Cg_in for grouped conv).
    const int kw = idx % kW;
    int t = idx / kW;
    const int kh = t % kH;
    t /= kH;
    const int c_in_local = t % Cg_in;
    const int c_out      = t / Cg_in;

    // Absolute input channel for this (c_out, c_in_local).
    const int g = c_out / Cg_out;
    const int c_in = g * Cg_in + c_in_local;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int i_out = 0; i_out < H_out; ++i_out) {
            const int in_h = i_out * stride_h - pad_h + kh * dil_h;
            if (in_h < 0 || in_h >= H) continue;
            for (int j_out = 0; j_out < W_out; ++j_out) {
                const int in_w = j_out * stride_w - pad_w + kw * dil_w;
                if (in_w < 0 || in_w >= W) continue;
                const int x_idx  = ((n * C_in  + c_in)  * H     + in_h)  * W     + in_w;
                const int dy_idx = ((n * C_out + c_out) * H_out + i_out) * W_out + j_out;
                acc += load_f32<T>(&dY[dy_idx]) * load_f32<T>(&X[x_idx]);
            }
        }
    }
    dWt_scratch[idx] = acc;
}

constexpr int BIAS_BLOCK = 256;

// One block per c_out; threads stride-loop over (n, i_out, j_out) summing
// dY[n, c_out, i_out, j_out] in FP32, then a shared-mem reduction folds the
// per-thread partials and thread 0 writes the per-channel sum into FP32
// scratch (one entry per c_out). The scratch is then folded into the
// caller's dB (storage-dtype-dispatched).
template <typename T>
__global__ void conv2d_backward_bias_kernel(
        const T* __restrict__ dY,
        float* __restrict__ dB_scratch,
        int N, int C_out, int H_out, int W_out) {
    const int c_out = blockIdx.x;
    const int tid = threadIdx.x;
    const int spatial = H_out * W_out;
    const int total_per_chan = N * spatial;

    float acc = 0.0f;
    for (int idx = tid; idx < total_per_chan; idx += blockDim.x) {
        const int n  = idx / spatial;
        const int sp = idx - n * spatial;
        const int dy_idx = (n * C_out + c_out) * spatial + sp;
        acc += load_f32<T>(&dY[dy_idx]);
    }

    __shared__ float s_acc[BIAS_BLOCK];
    s_acc[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_acc[tid] += s_acc[tid + stride];
        __syncthreads();
    }
    if (tid == 0) dB_scratch[c_out] = s_acc[0];
}

// Fold FP32 scratch into FP16/FP32 destination accumulators (add, not overwrite).
__global__ void conv2d_add_fp32_into_fp16(const float* __restrict__ src,
                                          __half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float prev = __half2float(dst[i]);
    dst[i] = __float2half(prev + src[i]);
}

__global__ void conv2d_add_fp32_into_fp32(const float* __restrict__ src,
                                          float* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] += src[i];
}

__global__ void conv2d_add_fp32_into_bf16(const float* __restrict__ src,
                                          __nv_bfloat16* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float prev = __bfloat162float(dst[i]);
    dst[i] = __float2bfloat16(prev + src[i]);
}

} // namespace

namespace detail::cuda {

// Pooled, stream-ordered scratch allocator (defined in tensor.cu). Backward
// scratch is allocated/freed through this instead of raw cudaMalloc/cudaFree so
// the per-op scratch sites in the inversion loop stop synchronizing the device.
void* cuda_alloc(std::size_t bytes);
void  cuda_free(void* ptr);

// conv2d_igemm_f32.cu: the FP32 implicit GEMM (groups == 1).
bool conv2d_igemm_f32(const float* X, const float* Wt, const float* bias, float* Y, int N,
                      int Cin, int H, int W, int Cout, int kH, int kW, int sh, int sw, int ph,
                      int pw, int dh, int dw, int Ho, int Wo, cudaStream_t stream);

// ops_conv_direct.cpp: MirKind::ConvDirect fallback.
void dispatch_conv_direct(std::uint32_t mode, Dtype dt,
                          const void* x, const void* w, void* y, const void* bias,
                          int N, int Cin, int T, int H, int Wd,
                          int Cout, int To, int Ho, int Wo,
                          int kT, int kH, int kW,
                          int st, int sh, int sw,
                          int pt, int ph, int pw,
                          int dt_, int dh, int dw, int groups);

void conv2d_forward(const ::brotensor::Tensor& X,
                    const ::brotensor::Tensor& Wt,
                    const ::brotensor::Tensor* bias,
                    int N, int C_in, int H, int W,
                    int C_out, int kH, int kW,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dil_h, int dil_w,
                    int groups,
                    ::brotensor::Tensor& Y) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 &&
        X.dtype != Dtype::BF16) {
        throw std::runtime_error("conv2d_forward: X must be FP16, BF16 or FP32");
    }
    if (Wt.dtype != X.dtype) {
        throw std::runtime_error("conv2d_forward: Wt dtype must match X");
    }
    if (bias && bias->dtype != X.dtype) {
        throw std::runtime_error("conv2d_forward: bias dtype must match X");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv2d_forward: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_forward: non-positive output shape");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    const int total = N * out_cols;
    if (total == 0) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());

    // ── 1x1 stride-1 pad-0: dispatch to the row-major GEMM ──
    //
    // With H_out == H and W_out == W the convolution reduces exactly to
    //     Y[n](C_out, H*W) = Wt(C_out, C_in) @ X[n](C_in, H*W),
    // and both operands are already contiguous in that layout — the implicit
    // GEMM's gather and scatter are addressing an im2col that is the identity.
    // The plain GEMM kernel is an order of magnitude faster on these shapes
    // (measured 8 -> 88 TFLOP/s at C=128, 256^2, batch 1), because a 1x1's K is
    // just C_in, so the conv kernel's fixed prologue and epilogue are amortised
    // over only a handful of K iterations.
    //
    // Bias is excluded rather than emulated: the GEMM has no bias epilogue, and
    // a separate add would give back what this saves. Every 1x1 in the diffusion
    // and terrain nets is bias-free (the converters fold it), so the exclusion
    // costs nothing in practice.
    //
    // The conv kernel does the whole batch in one launch; this does one launch
    // per batch item, because X's per-item (C_in, H*W) blocks are not
    // contiguous as a single (C_in, N*H*W) matrix. So the trade is a much
    // better kernel against N times the launch overhead, and it only pays when
    // each launch carries real work. Requiring a grid of at least 128 CTAs —
    // roughly one per SM on the cards this targets — admits the shapes where it
    // wins by an order of magnitude (C=128 at 256^2, batch 1: 8 -> 91 TFLOP/s)
    // and rejects the ones where it loses (C=768 at 8^2, batch 16, a 36-CTA
    // grid launched 16 times: measured 2.9x SLOWER than the conv kernel).
    constexpr int kMinGemmCtas = 128;
    if (groups == 1 && bias == nullptr && kH == 1 && kW == 1 &&
        pad_h == 0 && pad_w == 0 && stride_h == 1 && stride_w == 1 &&
        dil_h == 1 && dil_w == 1 &&
        (X.dtype == Dtype::FP16 || X.dtype == Dtype::BF16) &&
        detail::cuda::matmul_rm_internal::rm_wmma_grid_ctas(C_out, H * W) >=
            kMinGemmCtas) {
        const int HW = H * W;
        const std::size_t x_stride = static_cast<std::size_t>(C_in) * HW;
        const std::size_t y_stride = static_cast<std::size_t>(C_out) * HW;
        bool taken = true;
        for (int n = 0; n < N && taken; ++n) {
            if (X.dtype == Dtype::FP16) {
                taken = detail::cuda::matmul_rm_internal::launch_matmul_rm_wmma(
                    static_cast<const __half*>(Wt.data),
                    static_cast<const __half*>(X.data) + n * x_stride,
                    static_cast<__half*>(Y.data) + n * y_stride,
                    C_out, HW, C_in);
            } else {
                taken = detail::cuda::matmul_rm_internal::launch_matmul_rm_wmma_bf16(
                    static_cast<const __nv_bfloat16*>(Wt.data),
                    static_cast<const __nv_bfloat16*>(X.data) + n * x_stride,
                    static_cast<__nv_bfloat16*>(Y.data) + n * y_stride,
                    C_out, HW, C_in);
            }
        }
        // The gating is shape-only, so it either accepts every batch item or
        // rejects the first; a partial run would have written some of Y, but
        // the fall-through below overwrites all of it either way.
        if (taken) {
            BROTENSOR_CUDA_CHECK(cudaGetLastError());
            return;
        }
    }

    if (X.dtype == Dtype::FP16) {
        const __half* x_p  = static_cast<const __half*>(X.data);
        const __half* w_p  = static_cast<const __half*>(Wt.data);
        const __half* b_p  = bias ? static_cast<const __half*>(bias->data)
                                  : nullptr;
        __half* y_p        = static_cast<__half*>(Y.data);

        // Try the WMMA implicit-GEMM path for the SD1.5 + annotator shapes
        // (3x3 s1 p1, 1x1 s1 p0, 3x3 s2 p1, 3x3 s1 p0, 7x7 s1 p3,
        // 7x7 s1 p0, 5x5 s1 p2 — all dilation 1). The WMMA path assumes
        // full convolution (groups=1) — bypass it for grouped conv.
        if (groups == 1 &&
            conv2d_wmma_internal::launch_conv2d_implicit_gemm_wmma(
                x_p, w_p, b_p, y_p,
                N, C_in, H, W, C_out, kH, kW,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                H_out, W_out)) {
            BROTENSOR_CUDA_CHECK(cudaGetLastError());
            return;
        }
    } else if (X.dtype == Dtype::BF16) {
        const __nv_bfloat16* x_p = static_cast<const __nv_bfloat16*>(X.data);
        const __nv_bfloat16* w_p = static_cast<const __nv_bfloat16*>(Wt.data);
        const __nv_bfloat16* b_p = bias ? static_cast<const __nv_bfloat16*>(bias->data)
                                        : nullptr;
        __nv_bfloat16* y_p       = static_cast<__nv_bfloat16*>(Y.data);

        // BF16 WMMA implicit-GEMM fast path — same shape gating as FP16.
        if (groups == 1 &&
            conv2d_wmma_internal::launch_conv2d_implicit_gemm_wmma_bf16(
                x_p, w_p, b_p, y_p,
                N, C_in, H, W, C_out, kH, kW,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                H_out, W_out)) {
            BROTENSOR_CUDA_CHECK(cudaGetLastError());
            return;
        }
    } else {
        const float* x_p = static_cast<const float*>(X.data);
        const float* w_p = static_cast<const float*>(Wt.data);
        const float* b_p = bias ? static_cast<const float*>(bias->data) : nullptr;
        float* y_p       = static_cast<float*>(Y.data);
        if (groups == 1 &&
            conv2d_igemm_f32(x_p, w_p, b_p, y_p, N, C_in, H, W, C_out, kH, kW,
                             stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                             H_out, W_out, stream)) {
            return;
        }
    }
    dispatch_conv_direct(
        0u, X.dtype, X.data, Wt.data, Y.data, bias ? bias->data : nullptr,
        N, C_in, 1, H, W, C_out, 1, H_out, W_out, 1, kH, kW,
        1, stride_h, stride_w, 0, pad_h, pad_w, 1, dil_h, dil_w, groups);
}

void conv2d_backward_input(const ::brotensor::Tensor& Wt,
                           const ::brotensor::Tensor& dY,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dil_h, int dil_w,
                           int groups,
                           ::brotensor::Tensor& dX) {
    if (Wt.dtype != Dtype::FP16 && Wt.dtype != Dtype::FP32 &&
        Wt.dtype != Dtype::BF16) {
        throw std::runtime_error("conv2d_backward_input: Wt must be FP16, BF16 or FP32");
    }
    if (dY.dtype != Wt.dtype) {
        throw std::runtime_error("conv2d_backward_input: dY dtype must match Wt");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv2d_backward_input: groups must be >=1 and divide both C_in and C_out");
    }
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_backward_input: non-positive output shape");
    }
    const int in_cols = C_in * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != Wt.dtype) {
        dX.resize(N, in_cols, Wt.dtype);
    }
    if (N == 0 || in_cols == 0) return;

    dispatch_conv_direct(
        1u, Wt.dtype, dY.data, Wt.data, dX.data, nullptr,
        N, C_out, 1, H_out, W_out, C_in, 1, H, W, 1, kH, kW,
        1, stride_h, stride_w, 0, pad_h, pad_w, 1, dil_h, dil_w, groups);
}

void conv2d_backward_weight(const ::brotensor::Tensor& X,
                            const ::brotensor::Tensor& dY,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            int dil_h, int dil_w,
                            int groups,
                            ::brotensor::Tensor& dWt) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 &&
        X.dtype != Dtype::BF16) {
        throw std::runtime_error("conv2d_backward_weight: X must be FP16, BF16 or FP32");
    }
    if (dY.dtype != X.dtype || dWt.dtype != X.dtype) {
        throw std::runtime_error("conv2d_backward_weight: X, dY, dWt dtype must match");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv2d_backward_weight: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_backward_weight: non-positive output shape");
    }
    if (dWt.rows != C_out || dWt.cols != Cg_in * kH * kW) {
        throw std::runtime_error("conv2d_backward_weight: dWt shape mismatch");
    }
    const int total = C_out * Cg_in * kH * kW;
    if (total == 0) return;

    // FP32 scratch for the per-element partial sum. We write it (overwrite)
    // in the main kernel, then fold into the caller's accumulator (add). Pooled
    // + stream-ordered: cudaFreeAsync orders the reclaim after the fold kernel,
    // so no device sync is needed.
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    float* d_scratch = static_cast<float*>(cuda_alloc(total * sizeof(float)));

    // Small weight grids (depthwise / pointwise / small-channel convs) can't
    // fill the GPU with one thread per element — use one block per element
    // with the reduction parallelized across threads instead.
    const bool block_per_element = total < 65536;
    if (block_per_element) {
        if (X.dtype == Dtype::FP16) {
            conv2d_backward_weight_block_kernel<__half><<<total, CONV_BLOCK, 0, stream>>>(
                static_cast<const __half*>(X.data),
                static_cast<const __half*>(dY.data),
                d_scratch,
                N, C_in, H, W, C_out, kH, kW, H_out, W_out,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                groups, Cg_in, Cg_out);
        } else if (X.dtype == Dtype::BF16) {
            conv2d_backward_weight_block_kernel<__nv_bfloat16><<<total, CONV_BLOCK, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(X.data),
                static_cast<const __nv_bfloat16*>(dY.data),
                d_scratch,
                N, C_in, H, W, C_out, kH, kW, H_out, W_out,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                groups, Cg_in, Cg_out);
        } else {
            conv2d_backward_weight_block_kernel<float><<<total, CONV_BLOCK, 0, stream>>>(
                static_cast<const float*>(X.data),
                static_cast<const float*>(dY.data),
                d_scratch,
                N, C_in, H, W, C_out, kH, kW, H_out, W_out,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                groups, Cg_in, Cg_out);
        }
    } else {
        const int blocks = (total + CONV_BLOCK - 1) / CONV_BLOCK;
        if (X.dtype == Dtype::FP16) {
            conv2d_backward_weight_kernel<__half><<<blocks, CONV_BLOCK, 0, stream>>>(
                static_cast<const __half*>(X.data),
                static_cast<const __half*>(dY.data),
                d_scratch,
                N, C_in, H, W, C_out, kH, kW, H_out, W_out,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                groups, Cg_in, Cg_out, total);
        } else if (X.dtype == Dtype::BF16) {
            conv2d_backward_weight_kernel<__nv_bfloat16><<<blocks, CONV_BLOCK, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(X.data),
                static_cast<const __nv_bfloat16*>(dY.data),
                d_scratch,
                N, C_in, H, W, C_out, kH, kW, H_out, W_out,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                groups, Cg_in, Cg_out, total);
        } else {
            conv2d_backward_weight_kernel<float><<<blocks, CONV_BLOCK, 0, stream>>>(
                static_cast<const float*>(X.data),
                static_cast<const float*>(dY.data),
                d_scratch,
                N, C_in, H, W, C_out, kH, kW, H_out, W_out,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                groups, Cg_in, Cg_out, total);
        }
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    const int fold_blocks = (total + CONV_BLOCK - 1) / CONV_BLOCK;
    if (X.dtype == Dtype::FP16) {
        conv2d_add_fp32_into_fp16<<<fold_blocks, CONV_BLOCK, 0, stream>>>(
            d_scratch, static_cast<__half*>(dWt.data), total);
    } else if (X.dtype == Dtype::BF16) {
        conv2d_add_fp32_into_bf16<<<fold_blocks, CONV_BLOCK, 0, stream>>>(
            d_scratch, static_cast<__nv_bfloat16*>(dWt.data), total);
    } else {
        conv2d_add_fp32_into_fp32<<<fold_blocks, CONV_BLOCK, 0, stream>>>(
            d_scratch, static_cast<float*>(dWt.data), total);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    cuda_free(d_scratch);
}

void conv2d_backward_bias(const ::brotensor::Tensor& dY,
                          int N, int C_out, int H_out, int W_out,
                          ::brotensor::Tensor& dB) {
    if (dY.dtype != Dtype::FP16 && dY.dtype != Dtype::FP32 &&
        dY.dtype != Dtype::BF16) {
        throw std::runtime_error("conv2d_backward_bias: dY must be FP16, BF16 or FP32");
    }
    if (dB.dtype != dY.dtype) {
        throw std::runtime_error("conv2d_backward_bias: dB dtype must match dY");
    }
    if (dB.rows != C_out || dB.cols != 1) {
        throw std::runtime_error("conv2d_backward_bias: dB shape mismatch");
    }
    if (C_out == 0 || N == 0 || H_out == 0 || W_out == 0) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    float* d_scratch = static_cast<float*>(cuda_alloc(C_out * sizeof(float)));

    if (dY.dtype == Dtype::FP16) {
        conv2d_backward_bias_kernel<__half><<<C_out, BIAS_BLOCK, 0, stream>>>(
            static_cast<const __half*>(dY.data),
            d_scratch, N, C_out, H_out, W_out);
    } else if (dY.dtype == Dtype::BF16) {
        conv2d_backward_bias_kernel<__nv_bfloat16><<<C_out, BIAS_BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(dY.data),
            d_scratch, N, C_out, H_out, W_out);
    } else {
        conv2d_backward_bias_kernel<float><<<C_out, BIAS_BLOCK, 0, stream>>>(
            static_cast<const float*>(dY.data),
            d_scratch, N, C_out, H_out, W_out);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    const int fold_blocks = (C_out + 127) / 128;
    if (dY.dtype == Dtype::FP16) {
        conv2d_add_fp32_into_fp16<<<fold_blocks, 128, 0, stream>>>(
            d_scratch, static_cast<__half*>(dB.data), C_out);
    } else if (dY.dtype == Dtype::BF16) {
        conv2d_add_fp32_into_bf16<<<fold_blocks, 128, 0, stream>>>(
            d_scratch, static_cast<__nv_bfloat16*>(dB.data), C_out);
    } else {
        conv2d_add_fp32_into_fp32<<<fold_blocks, 128, 0, stream>>>(
            d_scratch, static_cast<float*>(dB.data), C_out);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    cuda_free(d_scratch);
}

} // namespace detail::cuda

namespace detail::cuda {

// Forward declarations of the resample ops defined in resample.cu — the
// linker resolves these at link time, allowing fill_cuda_vtable_conv below
// to wire them into the vtable without a public header.
void upsample_nearest_2x(const ::brotensor::Tensor& X,
                         int N, int C, int H, int W, ::brotensor::Tensor& Y);
void upsample_bilinear_2x(const ::brotensor::Tensor& X,
                          int N, int C, int H, int W, ::brotensor::Tensor& Y);
void downsample_avg_2x(const ::brotensor::Tensor& X,
                       int N, int C, int H, int W, ::brotensor::Tensor& Y);
void upsample_nearest_2x_backward(const ::brotensor::Tensor& dY,
                                  int N, int C, int H, int W, ::brotensor::Tensor& dX);
void upsample_bilinear_2x_backward(const ::brotensor::Tensor& dY,
                                   int N, int C, int H, int W, ::brotensor::Tensor& dX);
void downsample_avg_2x_backward(const ::brotensor::Tensor& dY,
                                int N, int C, int H, int W, ::brotensor::Tensor& dX);

// Per-cluster vtable contribution. Called from the aggregator that stitches
// together each cluster's slots into the CUDA OpsVTable and hands it to
// register_backend(Device::CUDA, ...).
void fill_cuda_vtable_conv(::brotensor::detail::OpsVTable& v) {
    v.conv2d_forward                = &conv2d_forward;
    v.conv2d_backward_input         = &conv2d_backward_input;
    v.conv2d_backward_weight        = &conv2d_backward_weight;
    v.conv2d_backward_bias          = &conv2d_backward_bias;
    // conv2d_int8w_fp16_forward lives in int8_quant.cu; its slot is filled
    // by that cluster's register helper, not here.
    v.upsample_nearest_2x           = &upsample_nearest_2x;
    v.upsample_bilinear_2x          = &upsample_bilinear_2x;
    v.downsample_avg_2x             = &downsample_avg_2x;
    v.upsample_nearest_2x_backward  = &upsample_nearest_2x_backward;
    v.upsample_bilinear_2x_backward = &upsample_bilinear_2x_backward;
    v.downsample_avg_2x_backward    = &downsample_avg_2x_backward;
}

} // namespace detail::cuda

} // namespace brotensor
