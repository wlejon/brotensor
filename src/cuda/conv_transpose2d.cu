// ─── CUDA 2D transposed convolution ─────────────────────────────────────────
//
// CUDA port of src/cpu/conv_transpose2d.cpp, mirroring the CPU contracts:
//   conv_transpose2d_forward / _backward_input / _backward_weight / _backward_bias
//
// The forward op is dtype-dispatched on X (FP32 / FP16 / BF16 — Wt and bias
// must match, Y resized to X's dtype; FP32 accumulation in the kernel, like
// conv2d). The three backward ops remain FP32-only.
//
// Layout (NCHW):
//   X / dX : (N, C_in*H*W)        — index ((n*C_in + c_in)*H + h)*W + w
//   Y / dY : (N, C_out*H_out*W_out)
//   Wt/dWt : (C_in, (C_out/groups)*kH*kW)  — input-channel-major
//            index (c_in*Cg_out + oc_local)*kHW + kh*kW + kw
//   bias   : (C_out, 1) or null
//
// Accumulation (matches the conv2d / conv_transpose1d contract):
//   *_forward / *_backward_input — output OVERWRITTEN.
//   _backward_weight / _bias     — dWt / dB ACCUMULATE (+=); caller zeros.
//
// Where the CPU forward op scatters, the CUDA forward op gathers (one thread
// per output element, no atomics) — the same linear map with a parallel-safe
// memory pattern. backward_weight pins one thread per weight element so no
// atomics are required either; backward_bias does a block-per-channel reduction.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor::detail::cuda {

namespace {

constexpr int CT2D_BLOCK = 256;

template <typename T>
__device__ inline float ct2d_load_f32(const T* p);
template <> __device__ inline float ct2d_load_f32<float>(const float* p)   { return *p; }
template <> __device__ inline float ct2d_load_f32<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float ct2d_load_f32<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }

template <typename T>
__device__ inline void ct2d_store_f32(T* p, float v);
template <> __device__ inline void ct2d_store_f32<float>(float* p, float v)   { *p = v; }
template <> __device__ inline void ct2d_store_f32<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void ct2d_store_f32<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
constexpr int CT2D_BIAS_BLOCK = 256;

inline int ct2d_grid(long long n) {
    long long blocks = (n + CT2D_BLOCK - 1) / CT2D_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void check_groups(const char* op, int C_in, int C_out, int groups) {
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        fail(op, "groups must be >=1 and divide both C_in and C_out");
    }
}

inline void require_fp32(const char* op, const ::brotensor::Tensor& t,
                         const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32 (CUDA conv_transpose2d is FP32-only)");
    }
}

inline int convt2d_out(int L, int stride, int padding, int output_padding,
                       int dilation, int kL) {
    return (L - 1) * stride - 2 * padding + dilation * (kL - 1)
           + output_padding + 1;
}

inline void check_geometry(const char* op, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int output_padding_h, int output_padding_w,
                           int dil_h, int dil_w) {
    if (kH < 1 || kW < 1 || stride_h < 1 || stride_w < 1
        || dil_h < 1 || dil_w < 1 || pad_h < 0 || pad_w < 0
        || output_padding_h < 0 || output_padding_w < 0) {
        fail(op, "kH/kW/stride/dilation >=1 and pad/output_padding >=0");
    }
    if (output_padding_h >= stride_h && output_padding_h >= dil_h) {
        fail(op, "output_padding_h must be < stride_h or < dil_h");
    }
    if (output_padding_w >= stride_w && output_padding_w >= dil_w) {
        fail(op, "output_padding_w must be < stride_w or < dil_w");
    }
}

// ─── forward — gather form ──────────────────────────────────────────────────
// One thread per (n, oc, oh, ow). CPU scatters input (n,c_in,h,w) to
//   ho = h*stride_h - pad_h + kh*dil_h
//   wo = w*stride_w - pad_w + kw*dil_w
// across every oc_local in c_in's group. Inverting: given (oh, ow),
//   h = (oh + pad_h - kh*dil_h) / stride_h  must be exact + in [0, H).
// Templated on storage dtype T (FP32 / FP16 / BF16); FP32 accumulator.
template <typename T>
__global__ void convt2d_forward_kernel(const T* __restrict__ X,
                                       const T* __restrict__ Wt,
                                       const T* __restrict__ bias,
                                       T* __restrict__ Y,
                                       int N, int C_in, int H, int W,
                                       int C_out, int kH, int kW,
                                       int stride_h, int stride_w,
                                       int pad_h, int pad_w,
                                       int dil_h, int dil_w,
                                       int Cg_in, int Cg_out,
                                       int H_out, int W_out) {
    const long long total = (long long)N * C_out * H_out * W_out;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int ow = static_cast<int>(idx % W_out);
        long long t = idx / W_out;
        const int oh = static_cast<int>(t % H_out);
        t /= H_out;
        const int oc = static_cast<int>(t % C_out);
        const int n  = static_cast<int>(t / C_out);

        const int g = oc / Cg_out;
        const int oc_local = oc - g * Cg_out;
        const int c_in_base = g * Cg_in;
        const int kHW = kH * kW;

        float acc = bias ? ct2d_load_f32<T>(&bias[oc]) : 0.0f;
        for (int kh = 0; kh < kH; ++kh) {
            const int num_h = oh + pad_h - kh * dil_h;
            if (num_h < 0 || num_h % stride_h != 0) continue;
            const int h = num_h / stride_h;
            if (h < 0 || h >= H) continue;
            for (int kw = 0; kw < kW; ++kw) {
                const int num_w = ow + pad_w - kw * dil_w;
                if (num_w < 0 || num_w % stride_w != 0) continue;
                const int w = num_w / stride_w;
                if (w < 0 || w >= W) continue;
                for (int ci = 0; ci < Cg_in; ++ci) {
                    const int c_in = c_in_base + ci;
                    const float xv = ct2d_load_f32<T>(
                        &X[((long long)n * C_in + c_in) * H * W + h * W + w]);
                    const float wv = ct2d_load_f32<T>(
                        &Wt[(long long)(c_in * Cg_out + oc_local) * kHW
                            + kh * kW + kw]);
                    acc += xv * wv;
                }
            }
        }
        ct2d_store_f32<T>(&Y[idx], acc);
    }
}

// ─── backward_input — gather (adjoint of the scatter is a plain conv) ───────
// One thread per (n, c_in, h, w). No atomics.
__global__ void convt2d_bwd_input_kernel(const float* __restrict__ Wt,
                                         const float* __restrict__ dY,
                                         float* __restrict__ dX,
                                         int N, int C_in, int H, int W,
                                         int C_out, int kH, int kW,
                                         int stride_h, int stride_w,
                                         int pad_h, int pad_w,
                                         int dil_h, int dil_w,
                                         int Cg_in, int Cg_out,
                                         int H_out, int W_out) {
    const long long total = (long long)N * C_in * H * W;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int w = static_cast<int>(idx % W);
        long long t = idx / W;
        const int h = static_cast<int>(t % H);
        t /= H;
        const int c_in = static_cast<int>(t % C_in);
        const int n    = static_cast<int>(t / C_in);

        const int g = c_in / Cg_in;
        const int oc_base = g * Cg_out;
        const int ho_origin = h * stride_h - pad_h;
        const int wo_origin = w * stride_w - pad_w;
        const int kHW = kH * kW;

        float acc = 0.0f;
        for (int kh = 0; kh < kH; ++kh) {
            const int ho = ho_origin + kh * dil_h;
            if (ho < 0 || ho >= H_out) continue;
            for (int kw = 0; kw < kW; ++kw) {
                const int wo = wo_origin + kw * dil_w;
                if (wo < 0 || wo >= W_out) continue;
                for (int oc_local = 0; oc_local < Cg_out; ++oc_local) {
                    const int oc = oc_base + oc_local;
                    const float wv =
                        Wt[(long long)(c_in * Cg_out + oc_local) * kHW
                           + kh * kW + kw];
                    const float gv =
                        dY[((long long)n * C_out + oc) * H_out * W_out
                           + ho * W_out + wo];
                    acc += gv * wv;
                }
            }
        }
        dX[idx] = acc;
    }
}

// ─── backward_weight ────────────────────────────────────────────────────────
// One thread per weight element (c_in, oc_local, kh, kw). Each thread owns a
// distinct weight slot — no atomics. Accumulates over (n, h, w) then += into
// dWt (caller zeroed it).
__global__ void convt2d_bwd_weight_kernel(const float* __restrict__ X,
                                          const float* __restrict__ dY,
                                          float* __restrict__ dWt,
                                          int N, int C_in, int H, int W,
                                          int C_out, int kH, int kW,
                                          int stride_h, int stride_w,
                                          int pad_h, int pad_w,
                                          int dil_h, int dil_w,
                                          int Cg_in, int Cg_out,
                                          int H_out, int W_out) {
    const long long total = (long long)C_in * Cg_out * kH * kW;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int kw = static_cast<int>(idx % kW);
        long long t = idx / kW;
        const int kh = static_cast<int>(t % kH);
        t /= kH;
        const int oc_local = static_cast<int>(t % Cg_out);
        const int c_in     = static_cast<int>(t / Cg_out);

        const int g = c_in / Cg_in;
        const int oc = g * Cg_out + oc_local;

        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            const float* x_chan =
                X + ((long long)n * C_in + c_in) * H * W;
            const float* dy_chan =
                dY + ((long long)n * C_out + oc) * H_out * W_out;
            for (int h = 0; h < H; ++h) {
                const int ho = h * stride_h - pad_h + kh * dil_h;
                if (ho < 0 || ho >= H_out) continue;
                for (int w = 0; w < W; ++w) {
                    const int wo = w * stride_w - pad_w + kw * dil_w;
                    if (wo < 0 || wo >= W_out) continue;
                    acc += x_chan[h * W + w] * dy_chan[ho * W_out + wo];
                }
            }
        }
        dWt[idx] += acc;
    }
}

// ─── backward_bias — block-per-channel reduction ────────────────────────────
// One block per c_out; threads stride-loop over (n, ho, wo), shared-mem reduce.
__global__ void convt2d_bwd_bias_kernel(const float* __restrict__ dY,
                                        float* __restrict__ dB,
                                        int N, int C_out,
                                        int H_out, int W_out) {
    const int oc = blockIdx.x;
    const int tid = threadIdx.x;
    const int spatial = H_out * W_out;
    const long long total_per_chan = (long long)N * spatial;

    float acc = 0.0f;
    for (long long i = tid; i < total_per_chan; i += blockDim.x) {
        const int n  = static_cast<int>(i / spatial);
        const int sp = static_cast<int>(i - (long long)n * spatial);
        const long long dy_idx =
            ((long long)n * C_out + oc) * spatial + sp;
        acc += dY[dy_idx];
    }

    __shared__ float s_acc[CT2D_BIAS_BLOCK];
    s_acc[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_acc[tid] += s_acc[tid + stride];
        __syncthreads();
    }
    if (tid == 0) dB[oc] += s_acc[0];
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════
//  Wrappers
// ════════════════════════════════════════════════════════════════════════════

void conv_transpose2d_forward(const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& Wt,
                              const ::brotensor::Tensor* bias,
                              int N, int C_in, int H, int W,
                              int C_out, int kH, int kW,
                              int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int output_padding_h, int output_padding_w,
                              int dil_h, int dil_w, int groups,
                              ::brotensor::Tensor& Y) {
    const char* op = "conv_transpose2d_forward";
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 &&
        X.dtype != Dtype::BF16) {
        fail(op, "X must be FP32, FP16 or BF16");
    }
    if (Wt.dtype != X.dtype) fail(op, "Wt dtype must match X");
    if (bias && bias->dtype != X.dtype) fail(op, "bias dtype must match X");
    check_groups(op, C_in, C_out, groups);
    check_geometry(op, kH, kW, stride_h, stride_w, pad_h, pad_w,
                   output_padding_h, output_padding_w, dil_h, dil_w);

    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = convt2d_out(H, stride_h, pad_h, output_padding_h,
                                  dil_h, kH);
    const int W_out = convt2d_out(W, stride_w, pad_w, output_padding_w,
                                  dil_w, kW);
    if (H_out <= 0 || W_out <= 0) fail(op, "non-positive output spatial size");

    const int kHW = kH * kW;
    if (Wt.rows != C_in || Wt.cols != Cg_out * kHW) {
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kH*kW)");
    }
    if (X.rows != N || X.cols != C_in * H * W) {
        fail(op, "X shape must be (N, C_in*H*W)");
    }
    if (bias && (bias->rows != C_out || bias->cols != 1)) {
        fail(op, "bias shape must be (C_out, 1)");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    if (N == 0 || out_cols == 0) return;

    const long long total = (long long)N * out_cols;
    if (X.dtype == Dtype::FP16) {
        convt2d_forward_kernel<__half><<<ct2d_grid(total), CT2D_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data),
            static_cast<const __half*>(Wt.data),
            bias ? static_cast<const __half*>(bias->data) : nullptr,
            static_cast<__half*>(Y.data),
            N, C_in, H, W, C_out, kH, kW,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            Cg_in, Cg_out, H_out, W_out);
    } else if (X.dtype == Dtype::BF16) {
        convt2d_forward_kernel<__nv_bfloat16><<<ct2d_grid(total), CT2D_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<const __nv_bfloat16*>(Wt.data),
            bias ? static_cast<const __nv_bfloat16*>(bias->data) : nullptr,
            static_cast<__nv_bfloat16*>(Y.data),
            N, C_in, H, W, C_out, kH, kW,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            Cg_in, Cg_out, H_out, W_out);
    } else {
        convt2d_forward_kernel<float><<<ct2d_grid(total), CT2D_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X.data),
            static_cast<const float*>(Wt.data),
            bias ? static_cast<const float*>(bias->data) : nullptr,
            static_cast<float*>(Y.data),
            N, C_in, H, W, C_out, kH, kW,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            Cg_in, Cg_out, H_out, W_out);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void conv_transpose2d_backward_input(const ::brotensor::Tensor& Wt,
                                     const ::brotensor::Tensor& dY,
                                     int N, int C_in, int H, int W,
                                     int C_out, int kH, int kW,
                                     int stride_h, int stride_w,
                                     int pad_h, int pad_w,
                                     int output_padding_h, int output_padding_w,
                                     int dil_h, int dil_w, int groups,
                                     ::brotensor::Tensor& dX) {
    const char* op = "conv_transpose2d_backward_input";
    require_fp32(op, Wt, "Wt");
    require_fp32(op, dY, "dY");
    check_groups(op, C_in, C_out, groups);
    check_geometry(op, kH, kW, stride_h, stride_w, pad_h, pad_w,
                   output_padding_h, output_padding_w, dil_h, dil_w);

    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = convt2d_out(H, stride_h, pad_h, output_padding_h,
                                  dil_h, kH);
    const int W_out = convt2d_out(W, stride_w, pad_w, output_padding_w,
                                  dil_w, kW);
    if (H_out <= 0 || W_out <= 0) fail(op, "non-positive output spatial size");
    const int kHW = kH * kW;
    if (Wt.rows != C_in || Wt.cols != Cg_out * kHW) {
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kH*kW)");
    }
    if (dY.rows != N || dY.cols != C_out * H_out * W_out) {
        fail(op, "dY shape must be (N, C_out*H_out*W_out)");
    }
    const int in_cols = C_in * H * W;
    if (dX.rows != N || dX.cols != in_cols
        || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(N, in_cols, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || in_cols == 0) return;

    const long long total = (long long)N * in_cols;
    convt2d_bwd_input_kernel<<<ct2d_grid(total), CT2D_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(Wt.data),
        static_cast<const float*>(dY.data),
        static_cast<float*>(dX.data),
        N, C_in, H, W, C_out, kH, kW,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        Cg_in, Cg_out, H_out, W_out);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void conv_transpose2d_backward_weight(const ::brotensor::Tensor& X,
                                      const ::brotensor::Tensor& dY,
                                      int N, int C_in, int H, int W,
                                      int C_out, int kH, int kW,
                                      int stride_h, int stride_w,
                                      int pad_h, int pad_w,
                                      int output_padding_h, int output_padding_w,
                                      int dil_h, int dil_w, int groups,
                                      ::brotensor::Tensor& dWt) {
    const char* op = "conv_transpose2d_backward_weight";
    require_fp32(op, X, "X");
    require_fp32(op, dY, "dY");
    require_fp32(op, dWt, "dWt");
    check_groups(op, C_in, C_out, groups);
    check_geometry(op, kH, kW, stride_h, stride_w, pad_h, pad_w,
                   output_padding_h, output_padding_w, dil_h, dil_w);

    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = convt2d_out(H, stride_h, pad_h, output_padding_h,
                                  dil_h, kH);
    const int W_out = convt2d_out(W, stride_w, pad_w, output_padding_w,
                                  dil_w, kW);
    if (H_out <= 0 || W_out <= 0) fail(op, "non-positive output spatial size");
    const int kHW = kH * kW;
    if (dWt.rows != C_in || dWt.cols != Cg_out * kHW) {
        fail(op, "dWt shape must be (C_in, (C_out/groups)*kH*kW)");
    }
    if (X.rows != N || X.cols != C_in * H * W) {
        fail(op, "X shape must be (N, C_in*H*W)");
    }
    if (dY.rows != N || dY.cols != C_out * H_out * W_out) {
        fail(op, "dY shape must be (N, C_out*H_out*W_out)");
    }
    if (C_in == 0 || Cg_out == 0 || kHW == 0) return;

    const long long total = (long long)C_in * Cg_out * kHW;
    convt2d_bwd_weight_kernel<<<ct2d_grid(total), CT2D_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(X.data),
        static_cast<const float*>(dY.data),
        static_cast<float*>(dWt.data),
        N, C_in, H, W, C_out, kH, kW,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        Cg_in, Cg_out, H_out, W_out);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void conv_transpose2d_backward_bias(const ::brotensor::Tensor& dY,
                                    int N, int C_out, int H_out, int W_out,
                                    ::brotensor::Tensor& dB) {
    const char* op = "conv_transpose2d_backward_bias";
    require_fp32(op, dY, "dY");
    require_fp32(op, dB, "dB");
    if (dB.rows != C_out || dB.cols != 1) {
        fail(op, "dB shape must be (C_out, 1)");
    }
    if (dY.rows != N || dY.cols != C_out * H_out * W_out) {
        fail(op, "dY shape must be (N, C_out*H_out*W_out)");
    }
    if (C_out == 0 || N == 0 || H_out == 0 || W_out == 0) return;

    convt2d_bwd_bias_kernel<<<C_out, CT2D_BIAS_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(dY.data),
        static_cast<float*>(dB.data),
        N, C_out, H_out, W_out);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_conv_transpose2d(::brotensor::detail::OpsVTable& v) {
    v.conv_transpose2d_forward         = &conv_transpose2d_forward;
    v.conv_transpose2d_backward_input  = &conv_transpose2d_backward_input;
    v.conv_transpose2d_backward_weight = &conv_transpose2d_backward_weight;
    v.conv_transpose2d_backward_bias   = &conv_transpose2d_backward_bias;
}

} // namespace brotensor::detail::cuda
