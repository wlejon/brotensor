// Native AMD HIP 2D transposed convolution (conv_transpose2d).
//
// Implements:
//   - conv_transpose2d_forward (FP32 / FP16 / BF16)
//   - conv_transpose2d_backward_input (FP32)
//   - conv_transpose2d_backward_weight (FP32, accumulates +=)
//   - conv_transpose2d_backward_bias (FP32, accumulates +=)
//
// Optimized for Wave32 wavefronts on RDNA 3.5.

#include "detail/conv_internal.h"
#include "detail/hip_check.h"

#include <brotensor/tensor.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);

namespace {

constexpr int CT2D_BLOCK = 256;

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

inline int ct2d_grid(long long n) {
    long long blocks = (n + CT2D_BLOCK - 1) / CT2D_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

template <typename T>
__device__ inline float ct2d_load_f32(const T* p);
template <> __device__ inline float ct2d_load_f32<float>(const float* p)   { return *p; }
template <> __device__ inline float ct2d_load_f32<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float ct2d_load_f32<hip_bfloat16>(const hip_bfloat16* p) { return float(*p); }

template <typename T>
__device__ inline void ct2d_store_f32(T* p, float v);
template <> __device__ inline void ct2d_store_f32<float>(float* p, float v)   { *p = v; }
template <> __device__ inline void ct2d_store_f32<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void ct2d_store_f32<hip_bfloat16>(hip_bfloat16* p, float v) { *p = hip_bfloat16(v); }

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
        fail(op, std::string(name) + " must be FP32 (HIP conv_transpose2d backward is FP32-only)");
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
    if ((output_padding_h >= stride_h && output_padding_h >= dil_h) ||
        (output_padding_w >= stride_w && output_padding_w >= dil_w)) {
        fail(op, "output_padding must be < stride or < dilation in each dim");
    }
}

// ─── forward kernel (gather) ────────────────────────────────────────────────
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
    const int kHW = kH * kW;
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

        float acc = bias ? ct2d_load_f32(&bias[oc]) : 0.0f;
        for (int kh = 0; kh < kH; ++kh) {
            const int num_h = oh + pad_h - kh * dil_h;
            if (num_h < 0 || num_h % stride_h != 0) continue;
            const int ih = num_h / stride_h;
            if (ih < 0 || ih >= H) continue;

            for (int kw = 0; kw < kW; ++kw) {
                const int num_w = ow + pad_w - kw * dil_w;
                if (num_w < 0 || num_w % stride_w != 0) continue;
                const int iw = num_w / stride_w;
                if (iw < 0 || iw >= W) continue;

                const int w_off_base = (oc_local * kH + kh) * kW + kw;
                for (int ci = 0; ci < Cg_in; ++ci) {
                    const int c_in = c_in_base + ci;
                    const long long x_idx =
                        (((long long)n * C_in + c_in) * H + ih) * W + iw;
                    const long long w_idx =
                        (long long)c_in * (Cg_out * kHW) + w_off_base;
                    acc += ct2d_load_f32(&X[x_idx]) * ct2d_load_f32(&Wt[w_idx]);
                }
            }
        }
        ct2d_store_f32(&Y[idx], acc);
    }
}

// ─── backward input kernel (gather) ─────────────────────────────────────────
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
    const int kHW = kH * kW;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int iw = static_cast<int>(idx % W);
        long long t = idx / W;
        const int ih = static_cast<int>(t % H);
        t /= H;
        const int c_in = static_cast<int>(t % C_in);
        const int n    = static_cast<int>(t / C_in);
        const int g = c_in / Cg_in;
        const int oc_base = g * Cg_out;
        const int oh_origin = ih * stride_h - pad_h;
        const int ow_origin = iw * stride_w - pad_w;

        float acc = 0.0f;
        for (int kh = 0; kh < kH; ++kh) {
            const int oh = oh_origin + kh * dil_h;
            if (oh < 0 || oh >= H_out) continue;
            for (int kw = 0; kw < kW; ++kw) {
                const int ow = ow_origin + kw * dil_w;
                if (ow < 0 || ow >= W_out) continue;

                const int w_off_base = (kh * kW + kw);
                for (int oc_local = 0; oc_local < Cg_out; ++oc_local) {
                    const int oc = oc_base + oc_local;
                    const long long w_idx =
                        (long long)c_in * (Cg_out * kHW)
                        + (long long)oc_local * kHW + w_off_base;
                    const long long dy_idx =
                        (((long long)n * C_out + oc) * H_out + oh) * W_out + ow;
                    acc += dY[dy_idx] * Wt[w_idx];
                }
            }
        }
        dX[idx] = acc;
    }
}

// ─── backward weight kernel (one thread per weight element) ─────────────────
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
    const int kHW = kH * kW;
    const long long total = (long long)C_in * Cg_out * kHW;
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
            const float* x_plane =
                X + ((long long)n * C_in + c_in) * H * W;
            const float* dy_plane =
                dY + ((long long)n * C_out + oc) * H_out * W_out;
            for (int ih = 0; ih < H; ++ih) {
                const int oh = ih * stride_h - pad_h + kh * dil_h;
                if (oh < 0 || oh >= H_out) continue;
                for (int iw = 0; iw < W; ++iw) {
                    const int ow = iw * stride_w - pad_w + kw * dil_w;
                    if (ow < 0 || ow >= W_out) continue;
                    acc += x_plane[ih * W + iw] * dy_plane[oh * W_out + ow];
                }
            }
        }
        dWt[idx] += acc;
    }
}

// ─── backward bias kernel ───────────────────────────────────────────────────
__global__ void convt2d_bwd_bias_kernel(const float* __restrict__ dY,
                                        float* __restrict__ dB,
                                        int N, int C_out, int H_out, int W_out) {
    const int hw_out = H_out * W_out;
    for (int oc = blockIdx.x * blockDim.x + threadIdx.x; oc < C_out;
         oc += blockDim.x * gridDim.x) {
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            const float* row =
                dY + ((long long)n * C_out + oc) * hw_out;
            for (int p = 0; p < hw_out; ++p) acc += row[p];
        }
        dB[oc] += acc;
    }
}

} // namespace

void conv_transpose2d_forward(const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& Wt,
                              const ::brotensor::Tensor* bias,
                              int N, int C_in, int H, int W, int C_out,
                              int kH, int kW, int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int output_padding_h, int output_padding_w,
                              int dil_h, int dil_w, int groups,
                              ::brotensor::Tensor& Y) {
    const char* op = "conv_transpose2d_forward";
    if (X.dtype != ::brotensor::Dtype::FP32 &&
        X.dtype != ::brotensor::Dtype::FP16 &&
        X.dtype != ::brotensor::Dtype::BF16) {
        fail(op, "X must be FP32, FP16, or BF16");
    }
    if (Wt.dtype != X.dtype) fail(op, "Wt dtype must match X");
    if (bias && bias->dtype != X.dtype) fail(op, "bias dtype must match X");
    check_groups(op, C_in, C_out, groups);
    check_geometry(op, kH, kW, stride_h, stride_w, pad_h, pad_w,
                   output_padding_h, output_padding_w, dil_h, dil_w);
    const int Cg_in = C_in / groups;
    const int Cg_out = C_out / groups;
    const int H_out = convt2d_out(H, stride_h, pad_h, output_padding_h, dil_h, kH);
    const int W_out = convt2d_out(W, stride_w, pad_w, output_padding_w, dil_w, kW);
    if (H_out <= 0 || W_out <= 0) fail(op, "non-positive output spatial shape");
    const int kHW = kH * kW;
    if (Wt.rows != C_in || Wt.cols != Cg_out * kHW) {
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kH*kW)");
    }
    if (bias && (bias->rows != C_out || bias->cols != 1)) {
        fail(op, "bias shape must be (C_out, 1)");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    if (N == 0 || C_out == 0 || H_out == 0 || W_out == 0) return;
    const long long total = (long long)N * C_out * H_out * W_out;
    hipStream_t stream = cur_stream();
    if (X.dtype == ::brotensor::Dtype::FP32) {
        const float* b_ptr = bias ? static_cast<const float*>(bias->data) : nullptr;
        convt2d_forward_kernel<float><<<ct2d_grid(total), CT2D_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data),
            static_cast<const float*>(Wt.data),
            b_ptr,
            static_cast<float*>(Y.data),
            N, C_in, H, W, C_out, kH, kW,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            Cg_in, Cg_out, H_out, W_out);
    } else if (X.dtype == ::brotensor::Dtype::FP16) {
        const __half* b_ptr = bias ? static_cast<const __half*>(bias->data) : nullptr;
        convt2d_forward_kernel<__half><<<ct2d_grid(total), CT2D_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data),
            static_cast<const __half*>(Wt.data),
            b_ptr,
            static_cast<__half*>(Y.data),
            N, C_in, H, W, C_out, kH, kW,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            Cg_in, Cg_out, H_out, W_out);
    } else {
        const hip_bfloat16* b_ptr = bias ? static_cast<const hip_bfloat16*>(bias->data) : nullptr;
        convt2d_forward_kernel<hip_bfloat16><<<ct2d_grid(total), CT2D_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data),
            static_cast<const hip_bfloat16*>(Wt.data),
            b_ptr,
            static_cast<hip_bfloat16*>(Y.data),
            N, C_in, H, W, C_out, kH, kW,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            Cg_in, Cg_out, H_out, W_out);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void conv_transpose2d_backward_input(const ::brotensor::Tensor& Wt,
                                     const ::brotensor::Tensor& dY,
                                     int N, int C_in, int H, int W, int C_out,
                                     int kH, int kW, int stride_h, int stride_w,
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
    const int Cg_in = C_in / groups;
    const int Cg_out = C_out / groups;
    const int H_out = convt2d_out(H, stride_h, pad_h, output_padding_h, dil_h, kH);
    const int W_out = convt2d_out(W, stride_w, pad_w, output_padding_w, dil_w, kW);
    if (H_out <= 0 || W_out <= 0) fail(op, "non-positive output spatial shape");
    const int kHW = kH * kW;
    if (Wt.rows != C_in || Wt.cols != Cg_out * kHW) {
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kH*kW)");
    }
    if (dY.rows != N || dY.cols != C_out * H_out * W_out) {
        fail(op, "dY shape must match (N, C_out*H_out*W_out)");
    }
    const int in_cols = C_in * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(N, in_cols, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || C_in == 0 || H == 0 || W == 0) return;
    const long long total = (long long)N * C_in * H * W;
    convt2d_bwd_input_kernel<<<ct2d_grid(total), CT2D_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(Wt.data),
        static_cast<const float*>(dY.data),
        static_cast<float*>(dX.data),
        N, C_in, H, W, C_out, kH, kW,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        Cg_in, Cg_out, H_out, W_out);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void conv_transpose2d_backward_weight(const ::brotensor::Tensor& X,
                                      const ::brotensor::Tensor& dY,
                                      int N, int C_in, int H, int W, int C_out,
                                      int kH, int kW, int stride_h, int stride_w,
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
    const int Cg_in = C_in / groups;
    const int Cg_out = C_out / groups;
    const int H_out = convt2d_out(H, stride_h, pad_h, output_padding_h, dil_h, kH);
    const int W_out = convt2d_out(W, stride_w, pad_w, output_padding_w, dil_w, kW);
    if (H_out <= 0 || W_out <= 0) fail(op, "non-positive output spatial shape");
    const int kHW = kH * kW;
    if (X.rows != N || X.cols != C_in * H * W) {
        fail(op, "X shape must match (N, C_in*H*W)");
    }
    if (dY.rows != N || dY.cols != C_out * H_out * W_out) {
        fail(op, "dY shape must match (N, C_out*H_out*W_out)");
    }
    if (dWt.rows != C_in || dWt.cols != Cg_out * kHW) {
        fail(op, "dWt shape mismatch");
    }
    if (N == 0 || C_in == 0 || C_out == 0) return;
    const long long total = (long long)C_in * Cg_out * kHW;
    convt2d_bwd_weight_kernel<<<ct2d_grid(total), CT2D_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(X.data),
        static_cast<const float*>(dY.data),
        static_cast<float*>(dWt.data),
        N, C_in, H, W, C_out, kH, kW,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        Cg_in, Cg_out, H_out, W_out);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void conv_transpose2d_backward_bias(const ::brotensor::Tensor& dY,
                                    int N, int C_out, int H_out, int W_out,
                                    ::brotensor::Tensor& dB) {
    const char* op = "conv_transpose2d_backward_bias";
    require_fp32(op, dY, "dY");
    require_fp32(op, dB, "dB");
    if (dY.rows != N || dY.cols != C_out * H_out * W_out) {
        fail(op, "dY shape must match (N, C_out*H_out*W_out)");
    }
    if (dB.rows != C_out || dB.cols != 1) {
        fail(op, "dB shape must be (C_out, 1)");
    }
    if (N == 0 || C_out == 0 || H_out == 0 || W_out == 0) return;
    convt2d_bwd_bias_kernel<<<ct2d_grid(C_out), CT2D_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(dY.data),
        static_cast<float*>(dB.data),
        N, C_out, H_out, W_out);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

} // namespace brotensor::detail::hip
